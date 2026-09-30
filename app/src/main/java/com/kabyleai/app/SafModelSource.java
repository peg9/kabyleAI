package com.kabyleai.app;

import android.content.ContentResolver;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.DocumentsContract;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Source basée sur le sélecteur de fichiers Android (Storage Access Framework) :
 * soit un dossier choisi (parcouru en profondeur, limitée), soit une liste de fichiers.
 */
public final class SafModelSource implements ModelSource {

    private static final int MAX_DEPTH = 4;
    private static final int MAX_ENTRIES = 5000;

    private final ContentResolver resolver;
    private final Uri tree;
    private final List<Uri> files;

    private SafModelSource(ContentResolver resolver, Uri tree, List<Uri> files) {
        this.resolver = resolver;
        this.tree = tree;
        this.files = files;
    }

    public static SafModelSource ofFolder(Context context, Uri tree) {
        return new SafModelSource(context.getContentResolver(), tree, null);
    }

    public static SafModelSource ofFiles(Context context, List<Uri> uris) {
        return new SafModelSource(context.getContentResolver(), null, new ArrayList<>(uris));
    }

    private static final class SafEntry implements Entry {
        private final ContentResolver resolver;
        private final Uri uri;
        private final String name;
        private final long size;

        SafEntry(ContentResolver resolver, Uri uri, String name, long size) {
            this.resolver = resolver;
            this.uri = uri;
            this.name = name;
            this.size = size;
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public long size() {
            return size;
        }

        @Override
        public InputStream open() throws IOException {
            InputStream stream = resolver.openInputStream(uri);
            if (stream == null) {
                throw new FileNotFoundException(name);
            }
            return stream;
        }
    }

    @Override
    public List<Entry> list() throws IOException {
        List<Entry> result = new ArrayList<>();
        if (files != null) {
            for (Uri uri : files) {
                addFromDocument(uri, result);
            }
        } else {
            String rootId = DocumentsContract.getTreeDocumentId(tree);
            List<String> level = new ArrayList<>();
            level.add(rootId);
            for (int depth = 0; depth <= MAX_DEPTH && !level.isEmpty(); depth++) {
                List<String> next = new ArrayList<>();
                for (String parent : level) {
                    walk(parent, result, next);
                }
                level = next;
            }
        }
        return result;
    }

    private void walk(String parentId, List<Entry> result, List<String> subFolders) throws IOException {
        Uri children = DocumentsContract.buildChildDocumentsUriUsingTree(tree, parentId);
        String[] columns = {
                DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                DocumentsContract.Document.COLUMN_MIME_TYPE,
                DocumentsContract.Document.COLUMN_SIZE
        };
        try (Cursor cursor = resolver.query(children, columns, null, null, null)) {
            if (cursor == null) {
                throw new IOException("dossier illisible");
            }
            while (cursor.moveToNext() && result.size() + subFolders.size() < MAX_ENTRIES) {
                String id = cursor.getString(0);
                String name = cursor.getString(1);
                String mime = cursor.getString(2);
                long size = cursor.isNull(3) ? -1 : cursor.getLong(3);
                if (DocumentsContract.Document.MIME_TYPE_DIR.equals(mime)) {
                    subFolders.add(id);
                } else if (name != null) {
                    Uri uri = DocumentsContract.buildDocumentUriUsingTree(tree, id);
                    result.add(new SafEntry(resolver, uri, name, size));
                }
            }
        } catch (SecurityException e) {
            throw new IOException("accès refusé au dossier", e);
        }
    }

    private void addFromDocument(Uri uri, List<Entry> result) throws IOException {
        // OpenableColumns fonctionne pour tout fournisseur ; les colonnes ci-dessus sont des synonymes.
        try (Cursor cursor = resolver.query(uri, new String[]{
                android.provider.OpenableColumns.DISPLAY_NAME,
                android.provider.OpenableColumns.SIZE}, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                String name = cursor.getString(0);
                long size = cursor.isNull(1) ? -1 : cursor.getLong(1);
                if (name != null) {
                    result.add(new SafEntry(resolver, uri, name, size));
                }
            }
        } catch (SecurityException e) {
            throw new IOException("accès refusé à un fichier", e);
        }
    }
}
