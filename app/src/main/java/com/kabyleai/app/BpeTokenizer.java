package com.kabyleai.app;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

/**
 * Tokenizer SentencePiece de type BPE (celui de NLLB-200 dans
 * transformers), sans dépendance native.
 *
 * Il reproduit le BPE de Hugging Face : le texte normalisé est découpé en
 * mots (chaque mot commence par "▁"), chaque mot en caractères, puis la
 * fusion de rang le plus bas est appliquée tant qu'il en existe une. Les
 * caractères absents du vocabulaire deviennent le jeton inconnu, les
 * inconnus consécutifs étant fusionnés.
 *
 * Fichiers écrits par export_nllb_onnx.py :
 *  - nllb_vocab.tsv : ligne i = "score TAB pièce" du jeton i ;
 *  - nllb_merges.txt : une fusion par ligne, "gauche ESPACE droite", par
 *    ordre de priorité.
 */
public final class BpeTokenizer implements Tokenizer {

    private static final char SPACE_MARK = '▁';
    private static final String UNKNOWN = "\u0000unk";
    private static final char PAIR_SEPARATOR = '\u0001';

    private final String[] pieces;
    private final boolean[] special;
    private final HashMap<String, Integer> index = new HashMap<>();
    private final HashMap<String, Integer> ranks = new HashMap<>();
    private final int unkId;
    private final boolean ignoreMerges;

    public BpeTokenizer(File vocabFile, File mergesFile, Set<Integer> specialIds,
                        int unkId, boolean ignoreMerges) throws IOException {
        this.unkId = unkId;
        this.ignoreMerges = ignoreMerges;

        List<String> pieceList = new ArrayList<>();
        try (BufferedReader reader = reader(vocabFile)) {
            String line;
            while ((line = reader.readLine()) != null) {
                int tab = line.indexOf('\t');
                if (tab < 0) {
                    throw new IOException("Ligne de vocabulaire invalide : " + line);
                }
                pieceList.add(line.substring(tab + 1));
            }
        }
        pieces = pieceList.toArray(new String[0]);
        special = new boolean[pieces.length];
        for (int i = 0; i < pieces.length; i++) {
            special[i] = specialIds.contains(i);
            if (!special[i]) {
                index.putIfAbsent(pieces[i], i);
            }
        }

        try (BufferedReader reader = reader(mergesFile)) {
            String line;
            int rank = 0;
            while ((line = reader.readLine()) != null) {
                int space = line.indexOf(' ');
                if (space <= 0) {
                    continue;
                }
                String left = line.substring(0, space);
                String right = line.substring(space + 1);
                ranks.putIfAbsent(left + PAIR_SEPARATOR + right, rank++);
            }
        }
    }

    private static BufferedReader reader(File file) throws IOException {
        return new BufferedReader(new InputStreamReader(
                new FileInputStream(file), StandardCharsets.UTF_8), 1 << 16);
    }

    @Override
    public int[] encode(String text) {
        int[] chars = UnigramTokenizer.normalize(text);
        if (chars.length == 0) {
            return new int[0];
        }

        List<Integer> ids = new ArrayList<>();
        int start = 0;
        for (int i = 1; i <= chars.length; i++) {
            if (i == chars.length || chars[i] == SPACE_MARK) {
                encodeWord(chars, start, i, ids);
                start = i;
            }
        }

        int[] result = new int[ids.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = ids.get(i);
        }
        return result;
    }

    private void encodeWord(int[] chars, int from, int to, List<Integer> out) {
        String word = new String(chars, from, to - from);
        if (ignoreMerges) {
            Integer whole = index.get(word);
            if (whole != null) {
                out.add(whole);
                return;
            }
        }

        List<String> symbols = new ArrayList<>();
        for (int i = from; i < to; i++) {
            String symbol = new String(chars, i, 1);
            if (index.containsKey(symbol)) {
                symbols.add(symbol);
            } else if (symbols.isEmpty() || !UNKNOWN.equals(symbols.get(symbols.size() - 1))) {
                symbols.add(UNKNOWN);
            }
        }

        while (symbols.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            int bestPosition = -1;
            for (int i = 0; i + 1 < symbols.size(); i++) {
                String left = symbols.get(i);
                String right = symbols.get(i + 1);
                if (UNKNOWN.equals(left) || UNKNOWN.equals(right)) {
                    continue;
                }
                Integer rank = ranks.get(left + PAIR_SEPARATOR + right);
                if (rank != null && rank < bestRank && index.containsKey(left + right)) {
                    bestRank = rank;
                    bestPosition = i;
                }
            }
            if (bestPosition < 0) {
                break;
            }
            symbols.set(bestPosition, symbols.get(bestPosition) + symbols.get(bestPosition + 1));
            symbols.remove(bestPosition + 1);
        }

        for (String symbol : symbols) {
            Integer id = UNKNOWN.equals(symbol) ? Integer.valueOf(unkId) : index.get(symbol);
            out.add(id == null ? unkId : id);
        }
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
        return out.toString().replace(SPACE_MARK, ' ').trim();
    }
}
