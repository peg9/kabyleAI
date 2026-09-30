# KabyleAI

Application Android pour le kabyle qui fonctionne sans connexion internet : traduire du français vers le kabyle, lire un texte kabyle à voix haute, transcrire du kabyle parlé. Les trois modèles tournent sur le téléphone avec ONNX Runtime.

## Fonctions

| Écran | Modèle | Rôle |
|---|---|---|
| Français → Kabyle | NLLB-200 (600M distillé) | Traduit un texte ; le bouton « Lire » envoie la traduction à Matoub, « Copier » la copie |
| Kabyle → Audio | Matoub-82M | Synthèse vocale, audio 24 kHz |
| Audio → Kabyle | Fadhma-300M | Reconnaissance vocale sur une fenêtre de 4 secondes |

## État du projet

- Développé et testé sur un seul téléphone Android rooté, avec une compilation faite directement sur le téléphone (Termux).
- La traduction, la synthèse vocale et la reconnaissance vocale fonctionnent de bout en bout sur cet appareil.
- Les procédures de compilation Linux et Windows suivent la marche habituelle d'un projet Android, mais elles n'ont pas été exécutées sur ce dépôt (voir [docs/COMPILER.md](docs/COMPILER.md)).

## Démarrage rapide

1. **Compiler l'application** : [docs/COMPILER.md](docs/COMPILER.md) (Linux, Windows, Termux).
2. **Préparer les modèles** : ils ne sont pas dans le dépôt, il faut les exporter au format ONNX. [docs/MODELES.md](docs/MODELES.md) décrit chaque export.
3. **Installer** l'APK et copier les modèles dans le dossier privé de l'application. Sur un téléphone rooté, `build_install.sh` fait les trois en une commande.

## Documentation

- [docs/COMPILER.md](docs/COMPILER.md) : compiler et installer l'application.
- [docs/MODELES.md](docs/MODELES.md) : exporter et installer les trois modèles.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) : fonctionnement interne et vérifications effectuées.

## Contenu du dépôt

```
app/                       Application Android (Kotlin, Java, Jetpack Compose)
  src/main/java/com/kabyleai/app/
    MainActivity.kt        Écran et reconnaissance vocale (Fadhma)
    MatoubTts.kt           Synthèse vocale (Matoub, deux graphes ONNX)
    MatoubTokenizer.kt     Phonétisation du kabyle pour Matoub
    TtsTextCleaner.java    Nettoyage du texte avant la synthèse
    NllbTranslator.java    Traduction (NLLB, trois graphes ONNX)
    BpeTokenizer.java      Tokenizer SentencePiece de type BPE
    UnigramTokenizer.java  Tokenizer SentencePiece de type Unigram
  src/main/assets/matoub/  vocab.json de Matoub
export_matoub_onnx.py      Export de Matoub-82M en deux graphes ONNX
export_nllb_onnx.py        Export de NLLB-200 en trois graphes ONNX
build_install.sh           Compile, installe et copie les modèles (Termux, root)
nllb_test.py               Test de qualité de NLLB (PyTorch, sans ONNX)
tools/nllb/                Vérification du tokenizer Java contre Hugging Face
docs/                      Documentation
```

Les autres scripts à la racine (`matoub_*.sh`, `test_matoub_*.py`, `connect_matoub_tts.sh` et voisins) sont des outils d'étape écrits pendant la mise au point. Ils sont conservés pour mémoire et ne servent pas à compiler ni à installer.

## Limites connues

- La qualité du kabyle produit par NLLB est inégale : la structure des phrases est souvent correcte, mais le vocabulaire peut être faux. L'écran affiche « à relire » pour cette raison.
- Le décodeur de traduction n'a pas de cache d'attention (KV cache) : il est rejoué en entier à chaque mot généré. Les phrases longues sont lentes.
- La reconnaissance vocale Fadhma travaille sur une fenêtre fixe de 4 secondes (64 000 échantillons à 16 kHz).
- Les chiffres et les symboles ne sont pas prononcés par la synthèse vocale : ils sont ignorés.
- Les modèles occupent plusieurs centaines de Mo, et l'export de NLLB demande de la mémoire vive et environ 5 Go de disque libre.

## Modèles et licences

Les modèles ne sont pas fournis avec ce dépôt. Vérifiez la licence de chacun auprès de sa source avant toute diffusion :

- NLLB-200 (Meta) : publié, à ma connaissance, sous licence CC-BY-NC 4.0, donc non commerciale. À confirmer sur la page du modèle.
- Matoub-82M et Fadhma-300M : voir la source d'où vous les tenez.
- ONNX Runtime (MIT) est utilisé par l'application ; PyTorch et Hugging Face Transformers (Apache 2.0) servent seulement à l'export.

Le code de ce dépôt n'a pas encore de licence : à choisir et à ajouter dans un fichier `LICENSE`.

## Pistes d'évolution

Ces points ne sont pas implémentés :

- Cache d'attention pour accélérer la traduction.
- Bouton « Enregistrer la correction » dans l'application, puis affinage de NLLB sur des paires français-kabyle corrigées.
- Wrapper Gradle commité, pour compiler avec `./gradlew` sur toutes les machines.
