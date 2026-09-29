package com.kabyleai.app;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

/**
 * Tokenizer SentencePiece de type Unigram (celui de NLLB-200), sans
 * dépendance native.
 *
 * Le vocabulaire est le fichier nllb_vocab.tsv écrit par export_nllb_onnx.py :
 * la ligne i contient "score TAB pièce" pour le jeton i.
 *
 * Normalisation : NFKC, espaces multiples réduits, suppression des
 * caractères de contrôle. C'est une approximation de la table "nmt_nfkc" de
 * SentencePiece ; tools/nllb/TokenizerCheck.java compare le résultat au
 * tokenizer Hugging Face.
 */
public final class UnigramTokenizer implements Tokenizer {

    private static final char SPACE_MARK = '▁';
    private static final float UNK_PENALTY = 10.0f;

    private final String[] pieces;
    private final float[] scores;
    private final boolean[] special;
    private final HashMap<String, Integer> index = new HashMap<>();
    private final int maxPieceLength;
    private final int unkId;
    private final float unkScore;

    public UnigramTokenizer(File vocabFile, Set<Integer> specialIds, int unkId)
            throws IOException {
        List<String> pieceList = new ArrayList<>();
        List<Float> scoreList = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(vocabFile), StandardCharsets.UTF_8), 1 << 16)) {
            String line;
            while ((line = reader.readLine()) != null) {
                int tab = line.indexOf('\t');
                if (tab < 0) {
                    throw new IOException("Ligne de vocabulaire invalide : " + line);
                }
                scoreList.add(Float.parseFloat(line.substring(0, tab)));
                pieceList.add(line.substring(tab + 1));
            }
        }

        int count = pieceList.size();
        pieces = pieceList.toArray(new String[0]);
        scores = new float[count];
        special = new boolean[count];
        this.unkId = unkId;

        float minScore = Float.MAX_VALUE;
        int longest = 1;
        for (int i = 0; i < count; i++) {
            scores[i] = scoreList.get(i);
            special[i] = specialIds.contains(i);
            if (special[i]) {
                continue;
            }
            index.putIfAbsent(pieces[i], i);
            minScore = Math.min(minScore, scores[i]);
            longest = Math.max(longest, pieces[i].codePointCount(0, pieces[i].length()));
        }
        maxPieceLength = longest;
        unkScore = minScore - UNK_PENALTY;
    }

    public int vocabularySize() {
        return pieces.length;
    }

    @Override
    public int[] encode(String text) {
        int[] chars = normalize(text);
        int n = chars.length;
        if (n == 0) {
            return new int[0];
        }

        float[] best = new float[n + 1];
        int[] bestId = new int[n + 1];
        int[] bestLength = new int[n + 1];
        java.util.Arrays.fill(best, Float.NEGATIVE_INFINITY);
        best[0] = 0.0f;

        for (int start = 0; start < n; start++) {
            if (best[start] == Float.NEGATIVE_INFINITY) {
                continue;
            }
            boolean hasSingle = false;
            int limit = Math.min(maxPieceLength, n - start);
            for (int length = 1; length <= limit; length++) {
                Integer id = index.get(new String(chars, start, length));
                if (id == null) {
                    continue;
                }
                if (length == 1) {
                    hasSingle = true;
                }
                float score = best[start] + scores[id];
                if (score > best[start + length]) {
                    best[start + length] = score;
                    bestId[start + length] = id;
                    bestLength[start + length] = length;
                }
            }
            if (!hasSingle) {
                float score = best[start] + unkScore;
                if (score > best[start + 1]) {
                    best[start + 1] = score;
                    bestId[start + 1] = unkId;
                    bestLength[start + 1] = 1;
                }
            }
        }

        ArrayList<Integer> reversed = new ArrayList<>();
        int position = n;
        while (position > 0) {
            reversed.add(bestId[position]);
            position -= bestLength[position];
        }

        // Comme le tokenizer Hugging Face : les jetons inconnus consécutifs
        // sont fusionnés en un seul.
        ArrayList<Integer> merged = new ArrayList<>();
        for (int i = reversed.size() - 1; i >= 0; i--) {
            int id = reversed.get(i);
            if (id == unkId && !merged.isEmpty() && merged.get(merged.size() - 1) == unkId) {
                continue;
            }
            merged.add(id);
        }

        int[] result = new int[merged.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = merged.get(i);
        }
        return result;
    }

    @Override
    public String decode(int[] ids) {
        StringBuilder out = new StringBuilder();
        for (int id : ids) {
            if (id < 0 || id >= pieces.length || special[id]) {
                continue;
            }
            out.append(pieces[id]);
        }
        String text = out.toString().replace(SPACE_MARK, ' ');
        return text.trim();
    }

    /** NFKC, espaces réduits, préfixe "▁" et espaces remplacés par "▁". */
    static int[] normalize(String text) {
        String nfkc = Normalizer.normalize(text, Normalizer.Form.NFKC);

        StringBuilder clean = new StringBuilder(nfkc.length() + 1);
        boolean previousSpace = true;
        for (int i = 0; i < nfkc.length(); ) {
            int cp = nfkc.codePointAt(i);
            i += Character.charCount(cp);

            // Comme la table nmt_nfkc de SentencePiece : ces caractères
            // valent une espace, les autres caractères de contrôle ASCII
            // sont supprimés. NFKC a déjà ramené les espaces Unicode à ' '.
            boolean whitespace = cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r'
                    || cp == 0x0C || cp == 0x200B || cp == 0x200C || cp == 0x200E
                    || cp == 0x200F || cp == 0x2028 || cp == 0x2029 || cp == 0xFEFF
                    || cp == 0xFFFD;
            if (whitespace) {
                if (!previousSpace) {
                    clean.append(' ');
                    previousSpace = true;
                }
                continue;
            }
            if (cp < 0x20 || cp == 0x7F) {
                continue;
            }
            clean.appendCodePoint(cp);
            previousSpace = false;
        }
        int end = clean.length();
        if (end > 0 && clean.charAt(end - 1) == ' ') {
            clean.setLength(end - 1);
        }
        if (clean.length() == 0) {
            return new int[0];
        }

        String marked = SPACE_MARK + clean.toString().replace(' ', SPACE_MARK);
        return marked.codePoints().toArray();
    }
}
