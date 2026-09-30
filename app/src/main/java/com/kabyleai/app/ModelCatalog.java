package com.kabyleai.app;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Liste des fichiers de modèles que l'application attend, et vérification de
 * ce qui est installé dans son dossier privé (filesDir).
 *
 * Aucune dépendance à Android : la classe se teste sur un simple dossier.
 */
public final class ModelCatalog {

    /** Un modèle, avec son sous-dossier de filesDir et son nom affiché. */
    public enum Group {
        MATOUB("matoub", "Matoub (synthèse vocale)", "Matoub"),
        NLLB("nllb", "NLLB (traduction)", "NLLB"),
        FADHMA("fadhma", "Fadhma (reconnaissance vocale)", "Fadhma");

        public final String directory;
        public final String label;
        public final String shortLabel;

        Group(String directory, String label, String shortLabel) {
            this.directory = directory;
            this.label = label;
            this.shortLabel = shortLabel;
        }
    }

    private static final String[] MATOUB_FILES = {
            "matoub_front.onnx", "matoub_back.onnx", "vocab.json"
    };

    private static final String[] NLLB_FILES = {
            "nllb_config.txt", "nllb_vocab.tsv", "nllb_embed.i8", "nllb_embed.scales",
            "nllb_encoder.onnx", "nllb_decoder.onnx", "nllb_head.onnx"
    };

    /** Nécessaire seulement si le tokenizer est de type BPE (celui de NLLB). */
    private static final String NLLB_MERGES = "nllb_merges.txt";

    private static final String[] FADHMA_FILES = {
            "fadhma_300m_prepared.onnx"
    };

    private static final Map<String, Group> KNOWN = new LinkedHashMap<>();

    static {
        for (String name : MATOUB_FILES) {
            KNOWN.put(name, Group.MATOUB);
        }
        for (String name : NLLB_FILES) {
            KNOWN.put(name, Group.NLLB);
        }
        KNOWN.put(NLLB_MERGES, Group.NLLB);
        for (String name : FADHMA_FILES) {
            KNOWN.put(name, Group.FADHMA);
        }
    }

    private ModelCatalog() {
    }

    /** Modèle auquel appartient ce nom de fichier, ou null s'il n'est pas reconnu. */
    public static Group groupOf(String fileName) {
        return fileName == null ? null : KNOWN.get(fileName);
    }

    /** Résultat de la vérification d'un modèle. */
    public static final class Status {
        public final Group group;
        public final List<String> missing = new ArrayList<>();
        public final List<String> problems = new ArrayList<>();
        /** Vrai si au moins un fichier de ce modèle existe. */
        public boolean anyPresent;

        Status(Group group) {
            this.group = group;
        }

        public boolean isComplete() {
            return missing.isEmpty() && problems.isEmpty();
        }

        /** "complet", "absent" ou "incomplet : ..." */
        public String describe() {
            if (isComplete()) {
                return "complet";
            }
            if (!anyPresent && problems.isEmpty()) {
                return "absent";
            }
            StringBuilder text = new StringBuilder("incomplet");
            if (!missing.isEmpty()) {
                text.append(", manque ").append(String.join(", ", missing));
            }
            if (!problems.isEmpty()) {
                text.append(", problème : ").append(String.join(" ; ", problems));
            }
            return text.toString();
        }
    }

    /** Vérifie un modèle dans filesDir. */
    public static Status check(Group group, File filesDir) {
        Status status = new Status(group);
        File directory = new File(filesDir, group.directory);

        List<String> required = new ArrayList<>();
        Map<String, String> config = new HashMap<>();
        switch (group) {
            case MATOUB:
                addAll(required, MATOUB_FILES);
                break;
            case FADHMA:
                addAll(required, FADHMA_FILES);
                break;
            default:
                addAll(required, NLLB_FILES);
                File configFile = new File(directory, "nllb_config.txt");
                if (configFile.isFile()) {
                    try {
                        config = readConfig(configFile);
                    } catch (IOException e) {
                        status.problems.add("nllb_config.txt illisible");
                    }
                    if ("bpe".equals(config.get("tokenizer"))) {
                        required.add(NLLB_MERGES);
                    }
                }
                break;
        }

        for (String name : required) {
            File file = new File(directory, name);
            if (!file.isFile()) {
                status.missing.add(name);
                continue;
            }
            status.anyPresent = true;
            if (file.length() == 0) {
                status.problems.add(name + " est vide");
            }
        }

        if (group == Group.NLLB) {
            checkNllbSizes(directory, config, status);
        }
        return status;
    }

    /** Les tables d'embeddings ont une taille imposée par la configuration. */
    private static void checkNllbSizes(File directory, Map<String, String> config, Status status) {
        long vocabSize;
        long dModel;
        try {
            vocabSize = Long.parseLong(config.getOrDefault("vocab_size", "-1"));
            dModel = Long.parseLong(config.getOrDefault("d_model", "-1"));
        } catch (NumberFormatException e) {
            status.problems.add("nllb_config.txt contient des valeurs invalides");
            return;
        }
        if (vocabSize <= 0 || dModel <= 0) {
            return;
        }
        File rows = new File(directory, "nllb_embed.i8");
        if (rows.isFile() && rows.length() != vocabSize * dModel) {
            status.problems.add("nllb_embed.i8 a une taille inattendue (copie interrompue ?)");
        }
        File scales = new File(directory, "nllb_embed.scales");
        if (scales.isFile() && scales.length() != 4L * vocabSize) {
            status.problems.add("nllb_embed.scales a une taille inattendue (copie interrompue ?)");
        }
    }

    private static void addAll(List<String> list, String[] names) {
        for (String name : names) {
            list.add(name);
        }
    }

    private static Map<String, String> readConfig(File file) throws IOException {
        Map<String, String> config = new HashMap<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                int eq = line.indexOf('=');
                if (eq > 0) {
                    config.put(line.substring(0, eq).trim(), line.substring(eq + 1).trim());
                }
            }
        }
        return config;
    }

    /** Une ligne par modèle, pour l'écran. */
    public static String summary(File filesDir) {
        StringBuilder text = new StringBuilder();
        for (Group group : Group.values()) {
            if (text.length() > 0) {
                text.append('\n');
            }
            text.append(group.label).append(" : ").append(check(group, filesDir).describe());
        }
        return text.toString();
    }

    /** Version courte, sur une ligne : "Matoub complet, NLLB absent, ...". */
    public static String shortSummary(File filesDir) {
        StringBuilder text = new StringBuilder();
        for (Group group : Group.values()) {
            if (text.length() > 0) {
                text.append(", ");
            }
            String state = check(group, filesDir).describe();
            int comma = state.indexOf(',');
            text.append(group.shortLabel).append(' ').append(comma < 0 ? state : state.substring(0, comma));
        }
        return text.toString();
    }
}
