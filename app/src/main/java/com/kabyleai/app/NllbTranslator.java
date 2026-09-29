package com.kabyleai.app;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Traduction français -> kabyle avec NLLB-200, hors ligne, par ONNX Runtime.
 *
 * Les fichiers viennent de export_nllb_onnx.py (voir ce script pour leur
 * contenu). Le décodage est glouton, sans cache clé/valeur : le décodeur est
 * rejoué à chaque pas et ne calcule les logits que pour la dernière position.
 */
public final class NllbTranslator implements AutoCloseable {

    /** Longueur maximale d'une phrase source, langue et fin comprises. */
    public static final int MAX_SOURCE_TOKENS = 400;
    /** Longueur maximale du décodeur (bornes de l'export ONNX). */
    private static final int MAX_TARGET_TOKENS = 400;

    private final File directory;
    private final OrtEnvironment environment = OrtEnvironment.getEnvironment();

    private Config config;
    private UnigramTokenizer tokenizer;
    private EmbeddingTable embeddings;
    private OrtSession encoder;
    private OrtSession decoder;
    private OrtSession head;

    /** Plafond de jetons générés (réglable pour les tests). */
    int targetCap = MAX_TARGET_TOKENS - 2;

    /** Interdit de répéter un groupe de n jetons déjà générés (0 = désactivé). */
    private final int noRepeatNgram;

    public NllbTranslator(File directory) {
        this(directory, 3);
    }

    public NllbTranslator(File directory, int noRepeatNgram) {
        this.directory = directory;
        this.noRepeatNgram = noRepeatNgram;
    }

    private static final class Config {
        int dModel;
        int vocabSize;
        int unkId;
        int eosId;
        int decoderStartId;
        int srcLangId;
        int tgtLangId;
        Set<Integer> specials = new HashSet<>();
    }

    private static Config readConfig(File file) throws IOException {
        Config config = new Config();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                int eq = line.indexOf('=');
                if (eq < 0) {
                    continue;
                }
                String key = line.substring(0, eq);
                String value = line.substring(eq + 1).trim();
                switch (key) {
                    case "d_model": config.dModel = Integer.parseInt(value); break;
                    case "vocab_size": config.vocabSize = Integer.parseInt(value); break;
                    case "unk_id": config.unkId = Integer.parseInt(value); break;
                    case "eos_id": config.eosId = Integer.parseInt(value); break;
                    case "decoder_start_id": config.decoderStartId = Integer.parseInt(value); break;
                    case "src_lang_id": config.srcLangId = Integer.parseInt(value); break;
                    case "tgt_lang_id": config.tgtLangId = Integer.parseInt(value); break;
                    case "specials":
                        for (String part : value.split(",")) {
                            if (!part.isEmpty()) {
                                config.specials.add(Integer.parseInt(part.trim()));
                            }
                        }
                        break;
                    default: break;
                }
            }
        }
        return config;
    }

    /** Matrice d'embeddings int8 (une échelle par ligne), lue par mappage mémoire. */
    private static final class EmbeddingTable implements AutoCloseable {
        private final int dModel;
        private final MappedByteBuffer rows;
        private final FloatBuffer scales;
        private final RandomAccessFile rowsFile;
        private final RandomAccessFile scalesFile;

        EmbeddingTable(File rowsPath, File scalesPath, int vocabSize, int dModel)
                throws IOException {
            this.dModel = dModel;
            rowsFile = new RandomAccessFile(rowsPath, "r");
            scalesFile = new RandomAccessFile(scalesPath, "r");
            if (rowsFile.length() != (long) vocabSize * dModel) {
                throw new IOException("nllb_embed.i8 : taille inattendue " + rowsFile.length());
            }
            if (scalesFile.length() != 4L * vocabSize) {
                throw new IOException("nllb_embed.scales : taille inattendue " + scalesFile.length());
            }
            rows = rowsFile.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, rowsFile.length());
            scales = scalesFile.getChannel()
                    .map(FileChannel.MapMode.READ_ONLY, 0, scalesFile.length())
                    .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        }

        /** Lignes demandées, à la suite, sous forme de tableau [n * d]. */
        float[] lookup(List<Integer> ids) {
            float[] out = new float[ids.size() * dModel];
            for (int i = 0; i < ids.size(); i++) {
                int id = ids.get(i);
                float scale = scales.get(id);
                int base = id * dModel;
                for (int j = 0; j < dModel; j++) {
                    out[i * dModel + j] = rows.get(base + j) * scale;
                }
            }
            return out;
        }

        @Override
        public void close() throws IOException {
            rowsFile.close();
            scalesFile.close();
        }
    }

    private synchronized void ensureLoaded() throws IOException, OrtException {
        if (encoder != null) {
            return;
        }
        config = readConfig(new File(directory, "nllb_config.txt"));
        tokenizer = new UnigramTokenizer(
                new File(directory, "nllb_vocab.tsv"), config.specials, config.unkId);
        embeddings = new EmbeddingTable(
                new File(directory, "nllb_embed.i8"),
                new File(directory, "nllb_embed.scales"),
                config.vocabSize, config.dModel);

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setIntraOpNumThreads(4);
        options.setInterOpNumThreads(1);
        encoder = environment.createSession(
                new File(directory, "nllb_encoder.onnx").getAbsolutePath(), options);
        decoder = environment.createSession(
                new File(directory, "nllb_decoder.onnx").getAbsolutePath(), options);
        head = environment.createSession(
                new File(directory, "nllb_head.onnx").getAbsolutePath(), options);
    }

    /** Vrai si tous les fichiers du modèle sont présents. */
    public boolean isInstalled() {
        String[] names = {"nllb_config.txt", "nllb_vocab.tsv", "nllb_embed.i8",
                "nllb_embed.scales", "nllb_encoder.onnx", "nllb_decoder.onnx",
                "nllb_head.onnx"};
        for (String name : names) {
            if (!new File(directory, name).exists()) {
                return false;
            }
        }
        return true;
    }

    /** Traduit un texte, phrase par phrase. */
    public String translate(String french) throws IOException, OrtException {
        if (!isInstalled()) {
            throw new IOException("Modèle de traduction introuvable dans " + directory);
        }
        ensureLoaded();

        StringBuilder result = new StringBuilder();
        for (String paragraph : french.split("\\R")) {
            if (paragraph.trim().isEmpty()) {
                result.append('\n');
                continue;
            }
            List<String> translated = new ArrayList<>();
            for (String sentence : splitSentences(paragraph)) {
                translated.add(translateSentence(sentence));
            }
            result.append(String.join(" ", translated)).append('\n');
        }
        return result.toString().trim();
    }

    /** Coupe un paragraphe après . ! ? … suivis d'un espace. */
    static List<String> splitSentences(String paragraph) {
        List<String> sentences = new ArrayList<>();
        for (String part : paragraph.trim().split("(?<=[.!?…])\\s+")) {
            if (!part.trim().isEmpty()) {
                sentences.add(part.trim());
            }
        }
        return sentences;
    }

    /** Traduit une phrase (entrée déjà découpée). */
    public String translateSentence(String sentence) throws IOException, OrtException {
        ensureLoaded();

        int[] body = tokenizer.encode(sentence);
        if (body.length == 0) {
            return "";
        }
        if (body.length + 2 > MAX_SOURCE_TOKENS) {
            throw new IllegalArgumentException(
                    "Phrase trop longue : " + body.length + " jetons, maximum "
                            + (MAX_SOURCE_TOKENS - 2) + ". Découpez-la.");
        }

        List<Integer> source = new ArrayList<>();
        source.add(config.srcLangId);
        for (int id : body) {
            source.add(id);
        }
        source.add(config.eosId);

        int[] output = generate(source);
        return tokenizer.decode(output);
    }

    /** Identifiants générés après la langue cible (fin de phrase exclue). */
    int[] generate(List<Integer> source) throws IOException, OrtException {
        ensureLoaded();
        int t = source.size();
        int d = config.dModel;

        long[] mask = new long[t];
        java.util.Arrays.fill(mask, 1L);

        try (OnnxTensor embeds = OnnxTensor.createTensor(
                     environment, FloatBuffer.wrap(embeddings.lookup(source)), new long[]{1, t, d});
             OnnxTensor maskTensor = OnnxTensor.createTensor(
                     environment, LongBuffer.wrap(mask), new long[]{1, t});
             OrtSession.Result encoded = encoder.run(encoderInputs(embeds, maskTensor))) {

            OnnxTensor hidden = (OnnxTensor) encoded.get(0);

            List<Integer> decoded = new ArrayList<>();
            decoded.add(config.decoderStartId);
            decoded.add(config.tgtLangId);

            int budget = Math.min(2 * t + 10, targetCap);
            for (int step = 0; step < budget; step++) {
                int s = decoded.size();
                Map<String, OnnxTensor> inputs = new HashMap<>();
                try (OnnxTensor stepEmbeds = OnnxTensor.createTensor(
                             environment, FloatBuffer.wrap(embeddings.lookup(decoded)),
                             new long[]{1, s, d})) {
                    inputs.put("decoder_embeds", stepEmbeds);
                    inputs.put("encoder_hidden_states", hidden);
                    inputs.put("encoder_attention_mask", maskTensor);

                    try (OrtSession.Result state = decoder.run(inputs);
                         OrtSession.Result scores = head.run(
                                 headInputs((OnnxTensor) state.get(0)))) {
                        float[] logits = ((float[][]) scores.get(0).getValue())[0];
                        int next = nextToken(logits, decoded);
                        decoded.add(next);
                        if (next == config.eosId) {
                            break;
                        }
                    }
                }
            }

            // On retire le jeton de départ, la langue forcée et la fin de phrase.
            List<Integer> text = decoded.subList(2, decoded.size());
            int end = text.size();
            if (end > 0 && text.get(end - 1) == config.eosId) {
                end--;
            }
            int[] ids = new int[end];
            for (int i = 0; i < end; i++) {
                ids[i] = text.get(i);
            }
            return ids;
        }
    }

    private static Map<String, OnnxTensor> headInputs(OnnxTensor hidden) {
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("hidden", hidden);
        return inputs;
    }

    private static Map<String, OnnxTensor> encoderInputs(OnnxTensor embeds, OnnxTensor mask) {
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("inputs_embeds", embeds);
        inputs.put("attention_mask", mask);
        return inputs;
    }

    /** Argmax, en interdisant les répétitions de n-grammes si demandé. */
    private int nextToken(float[] logits, List<Integer> decoded) {
        Set<Integer> banned = noRepeatNgram > 1 ? bannedTokens(decoded) : null;
        int best = -1;
        float bestValue = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.length; i++) {
            if (banned != null && banned.contains(i)) {
                continue;
            }
            if (logits[i] > bestValue) {
                bestValue = logits[i];
                best = i;
            }
        }
        return best;
    }

    /** Jetons qui recréeraient un n-gramme déjà présent (comme no_repeat_ngram_size). */
    private Set<Integer> bannedTokens(List<Integer> decoded) {
        Set<Integer> banned = new HashSet<>();
        int n = noRepeatNgram;
        int size = decoded.size();
        if (size + 1 < n) {
            return banned;
        }
        List<Integer> prefix = decoded.subList(size - (n - 1), size);
        for (int start = 0; start + n <= size; start++) {
            if (decoded.subList(start, start + n - 1).equals(prefix)) {
                banned.add(decoded.get(start + n - 1));
            }
        }
        return banned;
    }

    @Override
    public synchronized void close() throws IOException, OrtException {
        if (encoder != null) {
            encoder.close();
            decoder.close();
            head.close();
            embeddings.close();
            encoder = null;
            decoder = null;
        }
    }
}
