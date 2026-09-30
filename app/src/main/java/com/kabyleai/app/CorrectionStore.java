package com.kabyleai.app;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Paires (français, kabyle corrigé) enregistrées par l'utilisateur, pour un
 * affinage ultérieur du modèle de traduction.
 *
 * Fichier texte UTF-8, une paire par ligne, quatre colonnes séparées par une
 * tabulation : français, kabyle corrigé, kabyle proposé par le modèle, date.
 * Une même phrase française n'apparaît qu'une fois : une nouvelle correction
 * remplace l'ancienne. Aucune dépendance à Android.
 */
public final class CorrectionStore {

    public enum Result { ADDED, UPDATED, UNCHANGED, INVALID }

    private CorrectionStore() {
    }

    /** Une seule ligne, sans tabulation ni saut de ligne. */
    static String clean(String text) {
        return text == null ? "" : text.replaceAll("[\\t\\r\\n]+", " ").replaceAll(" {2,}", " ").trim();
    }

    public static synchronized Result save(File file, String french, String modelOutput, String corrected)
            throws IOException {
        String fr = clean(french);
        String kab = clean(corrected);
        if (fr.isEmpty() || kab.isEmpty()) {
            return Result.INVALID;
        }
        String model = clean(modelOutput);
        String date = new SimpleDateFormat("yyyy-MM-dd", Locale.ROOT).format(new Date());
        String line = fr + "\t" + kab + "\t" + model + "\t" + date;

        List<String> lines = readLines(file);
        Result result = Result.ADDED;
        for (int i = 0; i < lines.size(); i++) {
            String[] cols = lines.get(i).split("\t", -1);
            if (cols[0].equals(fr)) {
                if (cols.length > 1 && cols[1].equals(kab)) {
                    return Result.UNCHANGED;
                }
                lines.set(i, line);
                result = Result.UPDATED;
                break;
            }
        }
        if (result == Result.ADDED) {
            lines.add(line);
        }
        write(file, lines);
        return result;
    }

    public static synchronized int count(File file) {
        try {
            return readLines(file).size();
        } catch (IOException e) {
            return 0;
        }
    }

    /** Copie le fichier tel quel vers la destination choisie ; renvoie le nombre de paires. */
    public static synchronized int export(File file, OutputStream out) throws IOException {
        List<String> lines = readLines(file);
        StringBuilder text = new StringBuilder("francais\tkabyle_corrige\tkabyle_modele\tdate\n");
        for (String line : lines) {
            text.append(line).append('\n');
        }
        out.write(text.toString().getBytes(StandardCharsets.UTF_8));
        out.flush();
        return lines.size();
    }

    private static List<String> readLines(File file) throws IOException {
        List<String> lines = new ArrayList<>();
        if (!file.isFile()) {
            return lines;
        }
        try (InputStream in = new FileInputStream(file)) {
            String all = new String(readAll(in), StandardCharsets.UTF_8);
            for (String line : all.split("\n")) {
                if (!line.trim().isEmpty()) {
                    lines.add(line);
                }
            }
        }
        return lines;
    }

    private static byte[] readAll(InputStream in) throws IOException {
        java.io.ByteArrayOutputStream buffer = new java.io.ByteArrayOutputStream();
        byte[] chunk = new byte[8192];
        int read;
        while ((read = in.read(chunk)) > 0) {
            buffer.write(chunk, 0, read);
        }
        return buffer.toByteArray();
    }

    /** Écriture atomique : fichier provisoire puis renommage. */
    private static void write(File file, List<String> lines) throws IOException {
        File partial = new File(file.getPath() + ".part");
        StringBuilder text = new StringBuilder();
        for (String line : lines) {
            text.append(line).append('\n');
        }
        try (OutputStream out = new FileOutputStream(partial)) {
            out.write(text.toString().getBytes(StandardCharsets.UTF_8));
        }
        Files.move(partial.toPath(), file.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
    }
}
