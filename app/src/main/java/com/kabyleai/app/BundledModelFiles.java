package com.kabyleai.app;

import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/** Copie dans filesDir les petits fichiers livrés dans l'APK (assets). */
public final class BundledModelFiles {

    private BundledModelFiles() {
    }

    /** vocab.json de Matoub : livré dans l'APK, attendu dans filesDir/matoub. */
    public static void ensureMatoubVocab(Context context) {
        File target = new File(context.getFilesDir(), "matoub/vocab.json");
        if (target.isFile() && target.length() > 0) {
            return;
        }
        File directory = target.getParentFile();
        if (!directory.isDirectory() && !directory.mkdirs()) {
            return;
        }
        File partial = new File(directory, "vocab.json.part");
        try (InputStream in = context.getAssets().open("matoub/vocab.json");
             OutputStream out = new FileOutputStream(partial)) {
            byte[] buffer = new byte[16 * 1024];
            int read;
            while ((read = in.read(buffer)) > 0) {
                out.write(buffer, 0, read);
            }
        } catch (IOException e) {
            partial.delete();
            return;
        }
        if (!partial.renameTo(target)) {
            partial.delete();
        }
    }
}
