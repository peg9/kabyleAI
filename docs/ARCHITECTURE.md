# Architecture

Ce document décrit le fonctionnement interne de KabyleAI : ce que fait chaque partie, pourquoi les modèles sont découpés comme ils le sont, et ce qui a été vérifié.

## Vue d'ensemble

```
                      Texte français
                            |
                     [ NLLB-200 ]                 encodeur, décodeur, tête de sortie
                            |
                      Texte kabyle  <----------  saisie directe possible
                            |
                     [ TtsTextCleaner ]           retire ce que Matoub ne sait pas dire
                            |
                     [ MatoubTokenizer ]          phonétisation kabyle
                            |
                     [ Matoub-82M ]               deux graphes : durées, puis audio
                            |
                        Audio 24 kHz

   Micro (WAV 16 kHz, 4 s) --> [ Fadhma-300M ] --> Texte kabyle
```

Tous les modèles s'exécutent sur le téléphone avec ONNX Runtime (bibliothèque Android 1.30.0). Les calculs longs tournent dans des threads séparés, et les modèles sont chargés au premier usage.

## Synthèse vocale : Matoub-82M

Fichiers : `MatoubTts.kt`, `MatoubTokenizer.kt`, `TtsTextCleaner.java`.

1. **Nettoyage** (`TtsTextCleaner`) : normalisation Unicode, minuscules, `ε` et `γ` grecs ramenés à `ɛ` et `ɣ`, lettres accentuées du français ramenées à leur lettre de base, chiffres et symboles ignorés. Sans cela, `MatoubTokenizer` lève une exception dès qu'il rencontre un caractère inconnu.
2. **Phonétisation** (`MatoubTokenizer`) : le kabyle est converti en symboles phonétiques. Le code traite les consonnes emphatiques, la spirantisation de `b d g k t`, les géminées, l'assimilation des nasales et le timbre de `a` près des consonnes d'arrière (emphatiques, `q`, `ɣ`, `x`). Les identifiants sont encadrés par le symbole `$` à chaque bout.
3. **Premier graphe** (`matoub_front.onnx`) : prédit, pour chaque symbole, le nombre de trames audio.
4. **Alignement** : l'application construit la matrice symboles × trames (des 1 sur les trames de chaque symbole).
5. **Second graphe** (`matoub_back.onnx`) : reçoit les symboles, la sortie du premier graphe et l'alignement, et produit l'onde audio (24 kHz).
6. L'onde est écrite en WAV 16 bits et jouée avec `MediaPlayer`.

Limites de longueur : 510 symboles au plus (les deux `$` compris) et 20 000 trames au plus.

## Traduction : NLLB-200

Fichiers : `NllbTranslator.java`, `BpeTokenizer.java`, `UnigramTokenizer.java`, `Tokenizer.java`.

- **Découpage** : le texte est coupé en paragraphes, puis en phrases (après `.`, `!`, `?` ou `…` suivis d'un espace). Chaque phrase est traduite séparément et le résultat est recollé.
- **Tokenizer** : NLLB utilise un tokenizer SentencePiece. La version de Transformers utilisée le fournit de type BPE ; une variante Unigram est aussi gérée. Les deux sont réécrits en Java, sans bibliothèque native. Le vocabulaire et les fusions BPE viennent des fichiers `nllb_vocab.tsv` et `nllb_merges.txt`.
- **Embeddings** : la matrice des embeddings (256 206 lignes × 1 024) est stockée à part en int8 avec une échelle par ligne (`nllb_embed.i8`, `nllb_embed.scales`) et lue par mappage mémoire. L'application fait la consultation elle-même. Cela évite de dupliquer un gigaoctet de poids dans l'encodeur et le décodeur.
- **Trois graphes** : `nllb_encoder.onnx` (texte source), `nllb_decoder.onnx` (un pas de génération, résultat pour la dernière position seulement) et `nllb_head.onnx` (projection sur le vocabulaire). La tête est à part pour rester sous la limite de 2 Go des fichiers ONNX.
- **Génération** : décodage glouton. Le décodeur est rejoué en entier à chaque mot, sans cache d'attention. Les répétitions de groupes de 3 jetons sont interdites (équivalent de `no_repeat_ngram_size=3`). La langue cible `kab_Latn` est forcée comme premier jeton.
- **Limites** : 398 jetons source, 400 jetons de décodeur (bornes de l'export).

## Reconnaissance vocale : Fadhma-300M

Fichier : `MainActivity.kt`.

L'enregistrement est écrit en WAV 16 bits, mono, 16 kHz. Le modèle prend une fenêtre fixe de 64 000 échantillons (4 secondes) et produit des scores pour un vocabulaire de 40 symboles. Comme pour Matoub avant sa réécriture, la taille d'entrée figée est une limite connue.

## Pourquoi les exports sont découpés

Les modèles d'origine ne s'exportent pas tels quels en ONNX avec une longueur de texte variable. Quatre obstacles ont été rencontrés, et les scripts d'export les contournent :

1. **Nombres complexes.** `TorchSTFT` de Matoub utilise `torch.stft` et `torch.istft`, que l'export ONNX refuse. Le script les remplace par `RealSTFT`, qui calcule la même chose avec des convolutions réelles.
2. **Nombre de trames dépendant des données.** La somme des durées prédites n'est connue qu'à l'exécution, et `torch.export` ne peut pas s'en servir comme dimension. D'où l'export en deux graphes pour Matoub : l'alignement est construit entre les deux, dans l'application, ce qui en fait une dimension ordinaire du second graphe.
3. **Longueur figée par le LSTM.** À la trace, PyTorch exécute chaque LSTM avec la longueur de l'exemple et fige toutes les tailles calculées ensuite. Le modèle exporté ne marchait alors que pour cette longueur, ou donnait un audio faux. Les LSTM sont donc exportés comme opérateur ONNX `LSTM` (`ExportLSTM`, via `torch.onnx.ops.symbolic`).
4. **Formes périmées et poids derrière un `Transpose`.** Les formes gardées dans le graphe par l'export sont effacées et recalculées. Pour NLLB, les transpositions de poids sont repliées dans les constantes avant la quantification, sinon la quantification dynamique laisse ces couches en 32 bits.

## Vérifications effectuées

| Élément | Comment | Résultat |
|---|---|---|
| Tokenizers Java (Unigram, BPE) | Comparaison à SentencePiece sur des chaînes aléatoires (accents, espaces spéciaux, caractères rares) | 643 sur 643 (Unigram), 707 sur 707 (BPE) |
| Tokenizer BPE de NLLB | `TokenizerCheck` sur le vrai NLLB, sur téléphone | 26 sur 27, écart sans conséquence (espaces finaux) |
| Traducteur Java | ONNX Runtime pour la JVM sur un petit modèle M2M100 aléatoire de même structure, modèle complet quantifié | 22 générations sur 22 identiques à la référence |
| Export Matoub | Longueurs ONNX comparées à PyTorch pour 7 longueurs, sur le vrai modèle | Identiques |
| Export NLLB | Boucle gloutonne ONNX comparée à `generate()` de PyTorch, sur le vrai modèle | 6 sur 6 identiques |
| `TtsTextCleaner` | 5 000 chaînes aléatoires : la sortie ne contient que des caractères pris en charge | Vérifié |

Le petit modèle M2M100 aléatoire et le banc d'essai JVM ne sont pas dans le dépôt.

Ce qui n'a pas été vérifié : la compilation sur Linux et Windows, la qualité de la voix de Matoub mesurée autrement qu'à l'écoute, l'effet exact de la quantification int8 sur la qualité du kabyle de NLLB, et le temps de traduction sur d'autres téléphones que celui de l'auteur.

## Scripts historiques

À la racine, les scripts `matoub_dynamic.sh`, `matoub_dynamic_pipeline.sh`, `matoub_full_diagnostic.sh`, `connect_matoub_tts.sh`, `cleanup_kabyleai.sh`, `test_matoub_*.py`, `test_conv_export.py`, `test_onnx_patch.py`, `export_matoub_dynamic.py` et `export_matoub_legacy.py` ont servi à la mise au point de l'export Matoub. Les exports à utiliser aujourd'hui sont `export_matoub_onnx.py` et `export_nllb_onnx.py`.
