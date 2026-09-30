package com.kabyleai.app;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

/** Origine des fichiers à importer : un dossier ou une liste de fichiers choisis par l'utilisateur. */
public interface ModelSource {

    /** Un fichier disponible à l'import. */
    interface Entry {
        String name();

        /** Taille en octets, ou -1 si elle est inconnue. */
        long size();

        InputStream open() throws IOException;
    }

    /** Tous les fichiers disponibles, les moins profonds d'abord. */
    List<Entry> list() throws IOException;
}
