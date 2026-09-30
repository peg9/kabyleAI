package com.kabyleai.app;

import java.text.Normalizer;
import java.util.Locale;

/**
 * Prépare un texte pour la synthèse vocale Matoub.
 *
 * MatoubTokenizer refuse tout caractère qu'il ne sait pas prononcer (chiffres,
 * lettres accentuées du français, epsilon ou gamma grecs à la place de ɛ et ɣ,
 * que produit parfois la traduction). Ce nettoyage garde le texte lisible :
 *  - normalisation Unicode (ḍ écrit d + point souscrit devient ḍ) ;
 *  - minuscules ;
 *  - variantes courantes ramenées à l'alphabet kabyle (ε -> ɛ, γ -> ɣ) ;
 *  - lettres accentuées ramenées à leur lettre de base (é -> e) ;
 *  - tout le reste (chiffres, symboles, autres écritures) est ignoré.
 */
public final class TtsTextCleaner {

    /** Lettres que MatoubTokenizer sait prononcer. */
    private static final String LETTERS =
            "abcdefghijklmnopqrstuvwxyzčǧɛɣḍḥṛṣṭẓţ";

    /** Ponctuation que MatoubTokenizer retire lui-même. */
    private static final String PUNCTUATION =
            "«»\"'“”‘’.,;:!?()[]{}…";

    private TtsTextCleaner() {
    }

    /** Texte prêt pour MatoubTokenizer. */
    public static String clean(String text) {
        return process(text, null);
    }

    /** Caractères qui seront ignorés, séparés par des espaces (vide si aucun). */
    public static String ignoredCharacters(String text) {
        StringBuilder ignored = new StringBuilder();
        process(text, ignored);
        return ignored.toString();
    }

    private static String process(String text, StringBuilder ignored) {
        String prepared = Normalizer.normalize(text, Normalizer.Form.NFC)
                .toLowerCase(Locale.ROOT)
                .replace('ε', 'ɛ')
                .replace('γ', 'ɣ')
                .replace("œ", "oe")
                .replace("æ", "ae");

        StringBuilder out = new StringBuilder(prepared.length());
        for (int i = 0; i < prepared.length(); ) {
            int cp = prepared.codePointAt(i);
            i += Character.charCount(cp);

            if (Character.isWhitespace(cp) || cp == '-' || cp == '‑' || cp == '–') {
                out.append(cp == '-' ? '-' : ' ');
            } else if (cp < 0x10000 && (LETTERS.indexOf(cp) >= 0 || PUNCTUATION.indexOf(cp) >= 0)) {
                out.append((char) cp);
            } else {
                String base = baseLetter(new String(Character.toChars(cp)));
                if (base != null) {
                    out.append(base);
                } else if (ignored != null) {
                    if (ignored.length() > 0) {
                        ignored.append(' ');
                    }
                    ignored.appendCodePoint(cp);
                }
            }
        }
        return out.toString();
    }

    /** Lettre latine de base d'un caractère accentué, ou null. */
    private static String baseLetter(String character) {
        String decomposed = Normalizer.normalize(character, Normalizer.Form.NFD);
        StringBuilder base = new StringBuilder();
        for (int i = 0; i < decomposed.length(); i++) {
            char c = decomposed.charAt(i);
            if (Character.getType(c) != Character.NON_SPACING_MARK) {
                base.append(c);
            }
        }
        if (base.length() == 1 && base.charAt(0) >= 'a' && base.charAt(0) <= 'z') {
            return base.toString();
        }
        return null;
    }
}
