package com.kabyleai.app;

/** Tokenizer SentencePiece de NLLB : texte <-> identifiants de jetons. */
public interface Tokenizer {

    /** Texte -> identifiants de jetons, sans jetons spéciaux. */
    int[] encode(String text);

    /** Identifiants -> texte ; les jetons spéciaux sont ignorés. */
    String decode(int[] ids);
}
