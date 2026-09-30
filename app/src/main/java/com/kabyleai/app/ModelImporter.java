package com.kabyleai.app;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Copie les fichiers de modèles choisis par l'utilisateur vers le dossier
 * privé de l'application (filesDir/matoub, nllb, fadhma), avec progression.
 *
 * Chaque fichier est d'abord écrit sous "nom.part" puis renommé : une copie
 * interrompue ne laisse jamais un fichier tronqué sous son vrai nom.
 * Aucune dépendance à Android.
 */
public final class ModelImporter {

    /** Marge de place libre exigée en plus des fichiers à copier. */
    static final long SPACE_MARGIN = 64L * 1024 * 1024;

    private static final int BUFFER_SIZE = 1024 * 1024;
    private static final long PROGRESS_INTERVAL_MS = 200;

    public interface ProgressListener {
        /** fraction entre 0 et 1 ; message prêt à afficher. Appelé depuis le thread de copie. */
        void onProgress(float fraction, String message);
    }

    public static final class Report {
        public int copied;
        public long bytes;
        public int ignored;
        public int duplicates;
        public boolean cancelled;
        public boolean failed;
        public String message = "";
    }

    private ModelImporter() {
    }

    public static Report importFrom(ModelSource source, File filesDir,
                                    ProgressListener listener, AtomicBoolean cancel) {
        return importFrom(source, filesDir, listener, cancel, filesDir.getUsableSpace());
    }

    static Report importFrom(ModelSource source, File filesDir, ProgressListener listener,
                             AtomicBoolean cancel, long usableSpace) {
        Report report = new Report();
        List<ModelSource.Entry> entries;
        try {
            entries = source.list();
        } catch (IOException | RuntimeException e) {
            return fail(report, "Lecture de la sélection impossible : " + e.getMessage());
        }

        List<ModelSource.Entry> chosen = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        for (ModelSource.Entry entry : entries) {
            if (ModelCatalog.groupOf(entry.name()) == null) {
                report.ignored++;
            } else if (!seen.add(entry.name())) {
                report.duplicates++;
            } else {
                chosen.add(entry);
            }
        }

        if (chosen.isEmpty()) {
            return fail(report, "Aucun fichier de modèle reconnu dans la sélection ("
                    + entries.size() + " fichier(s) examiné(s)). Choisissez le dossier qui "
                    + "contient nllb_encoder.onnx, matoub_front.onnx, etc.");
        }

        long total = 0;
        boolean unknownSize = false;
        for (ModelSource.Entry entry : chosen) {
            if (entry.size() < 0) {
                unknownSize = true;
            } else {
                total += entry.size();
            }
        }

        if (!unknownSize && usableSpace < total + SPACE_MARGIN) {
            return fail(report, "Place insuffisante : il faut " + human(total + SPACE_MARGIN)
                    + " libres, il en reste " + human(usableSpace) + ".");
        }

        long done = 0;
        int index = 0;
        long lastReport = 0;
        byte[] buffer = new byte[BUFFER_SIZE];

        for (ModelSource.Entry entry : chosen) {
            index++;
            ModelCatalog.Group group = ModelCatalog.groupOf(entry.name());
            File directory = new File(filesDir, group.directory);
            if (!directory.isDirectory() && !directory.mkdirs()) {
                return fail(report, "Impossible de créer " + directory);
            }
            File target = new File(directory, entry.name());
            File partial = new File(directory, entry.name() + ".part");
            String label = "Copie " + index + "/" + chosen.size() + " : " + entry.name();

            long copiedHere = 0;
            try (InputStream in = entry.open(); OutputStream out = new FileOutputStream(partial)) {
                if (in == null) {
                    throw new IOException("fichier illisible");
                }
                int read;
                while ((read = in.read(buffer)) > 0) {
                    if (cancel != null && cancel.get()) {
                        report.cancelled = true;
                        break;
                    }
                    out.write(buffer, 0, read);
                    copiedHere += read;
                    long now = System.currentTimeMillis();
                    if (listener != null && now - lastReport >= PROGRESS_INTERVAL_MS) {
                        lastReport = now;
                        listener.onProgress(fraction(done + copiedHere, total, unknownSize),
                                label + " (" + human(copiedHere)
                                        + (entry.size() >= 0 ? " / " + human(entry.size()) : "") + ")");
                    }
                }
                out.flush();
            } catch (IOException | RuntimeException e) {
                partial.delete();
                return fail(report, "Échec de la copie de " + entry.name() + " : " + e.getMessage()
                        + reasonHint(e));
            }

            if (report.cancelled) {
                partial.delete();
                report.message = "Import annulé (" + report.copied + " fichier(s) déjà copié(s)).";
                return report;
            }

            if (entry.size() >= 0 && copiedHere != entry.size()) {
                partial.delete();
                return fail(report, entry.name() + " : " + copiedHere + " octets copiés sur "
                        + entry.size() + " attendus.");
            }

            if (target.exists() && !target.delete()) {
                partial.delete();
                return fail(report, "Impossible de remplacer " + target.getName());
            }
            if (!partial.renameTo(target)) {
                partial.delete();
                return fail(report, "Impossible de finaliser " + target.getName());
            }

            done += copiedHere;
            report.copied++;
            report.bytes += copiedHere;
        }

        if (listener != null) {
            listener.onProgress(1f, "Copie terminée, vérification...");
        }
        report.message = "Import terminé : " + report.copied + " fichier(s), " + human(report.bytes) + ".";
        if (report.duplicates > 0) {
            report.message += " " + report.duplicates + " doublon(s) ignoré(s).";
        }
        return report;
    }

    private static String reasonHint(Exception e) {
        String text = String.valueOf(e.getMessage()).toLowerCase(Locale.ROOT);
        return text.contains("space") || text.contains("enospc") ? " (espace disque épuisé)" : "";
    }

    private static Report fail(Report report, String message) {
        report.failed = true;
        report.message = message;
        return report;
    }

    private static float fraction(long done, long total, boolean unknownSize) {
        if (unknownSize || total <= 0) {
            return 0f;
        }
        return Math.min(1f, (float) ((double) done / total));
    }

    /** "1,3 Go", "412 Mo", "8 Ko". */
    public static String human(long bytes) {
        if (bytes >= 1L << 30) {
            return String.format(Locale.FRANCE, "%.2f Go", bytes / (double) (1L << 30));
        }
        if (bytes >= 1L << 20) {
            return String.format(Locale.FRANCE, "%.0f Mo", bytes / (double) (1L << 20));
        }
        if (bytes >= 1L << 10) {
            return String.format(Locale.FRANCE, "%.0f Ko", bytes / (double) (1L << 10));
        }
        return bytes + " o";
    }
}
