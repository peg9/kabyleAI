# Préparer et installer les modèles

Les modèles ne sont pas dans le dépôt. Chacun doit être exporté au format ONNX, puis copié dans le dossier privé de l'application sur le téléphone.

## Vue d'ensemble

| Fonction | Modèle | Script d'export | Dossier sur le téléphone |
|---|---|---|---|
| Kabyle → Audio | Matoub-82M | `export_matoub_onnx.py` | `files/matoub/` |
| Français → Kabyle | NLLB-200 distillé 600M | `export_nllb_onnx.py` | `files/nllb/` |
| Audio → Kabyle | Fadhma-300M | non fourni | `files/fadhma/` |

`files/` désigne `/data/data/com.kabyleai.app/files/`. L'application y lit les modèles avec `File(filesDir, ...)` : c'est un dossier privé, accessible avec root, ou avec `adb run-as` sur une version de débogage.

Fichiers attendus :

```
files/matoub/matoub_front.onnx
files/matoub/matoub_back.onnx
files/matoub/vocab.json              (copie de app/src/main/assets/matoub/vocab.json)

files/nllb/nllb_config.txt
files/nllb/nllb_vocab.tsv
files/nllb/nllb_merges.txt           (seulement si le tokenizer est de type BPE, ce qui est le cas de NLLB)
files/nllb/nllb_embed.i8
files/nllb/nllb_embed.scales
files/nllb/nllb_encoder.onnx
files/nllb/nllb_decoder.onnx
files/nllb/nllb_head.onnx

files/fadhma/fadhma_300m_prepared.onnx
```

L'application ne lit pas `vocab.json` depuis ses ressources internes : il faut le copier dans `files/matoub/`.

## Environnement Python pour les exports

Les exports demandent Python 3.11 ou plus récent avec :

```
pip install torch onnx onnxruntime onnxscript safetensors transformers numpy
```

Versions utilisées pendant la mise au point : PyTorch 2.14, ONNX Runtime 1.30, Transformers 5.x. Les exports utilisent `torch.onnx.export(..., dynamo=True)` : une version plus ancienne de PyTorch peut ne pas fonctionner.

Trois façons de disposer de cet environnement :

- **Un PC** (Linux, Windows, macOS) avec un environnement virtuel Python. Le chemin le plus simple.
- **Google Colab** ou un service équivalent, si le PC est trop juste.
- **Termux sur le téléphone**, mais pas avec le Python de Termux : la version de PyTorch de Termux plante pendant l'export (arrêt brutal avec le message `Pointer tag ... truncated`). Il faut passer par un Ubuntu à l'intérieur de Termux :

```bash
pkg install -y proot-distro
proot-distro install ubuntu
proot-distro login ubuntu -- bash -c "apt-get update && apt-get install -y python3 python3-pip python3-venv"
proot-distro login ubuntu -- bash -c "python3 -m venv /opt/v"
proot-distro login ubuntu -- /opt/v/bin/pip install torch onnx onnxruntime onnxscript safetensors transformers numpy
```

Les scripts s'exécutent ensuite avec, par exemple :

```bash
proot-distro login ubuntu \
  --bind $HOME/kabyle-models:/models \
  --bind $HOME/projects/KabyleAI:/work -- \
  bash -c "cd /work && MATOUB_MODEL=/models/matoub-model MATOUB_OUT_DIR=/models/matoub_dynamic /opt/v/bin/python export_matoub_onnx.py"
```

Prévoyez plusieurs Go de disque pour cet environnement (PyTorch installe aussi des paquets liés aux cartes graphiques Nvidia, inutiles ici).

## Matoub-82M (synthèse vocale)

### Ce qu'il faut avoir

Le script attend un dossier `matoub-model` (avec `config.json`, `model.safetensors`, `vocab.json`) et, à côté, le paquet Python `matoub_model` (`configuration_matoub.py`, `modeling_matoub.py`, `istftnet.py`) :

```
~/kabyle-models/
  matoub-model/
  matoub_model/
```

Ces fichiers ne sont pas dans le dépôt. D'après l'en-tête de `istftnet.py`, le code du modèle est adapté de Kokoro-82M (Apache 2.0) et de StyleTTS2 (MIT). La source des poids est à indiquer ici par l'auteur.

### Export

```bash
MATOUB_MODEL=~/kabyle-models/matoub-model \
MATOUB_OUT_DIR=~/kabyle-models/matoub_dynamic \
python export_matoub_onnx.py
```

Sous PowerShell : `$env:MATOUB_MODEL = "C:\chemin\matoub-model"; python export_matoub_onnx.py`.

Valeurs par défaut : `MATOUB_MODEL=~/kabyle-models/matoub-model` et `MATOUB_OUT_DIR=~/kabyle-models/matoub_dynamic`.

### Résultat attendu

Le script écrit `matoub_front.onnx` et `matoub_back.onnx`, puis compare leur sortie à PyTorch pour sept longueurs de texte. Il faut voir :

- `6 LSTM remplacés.` (le nombre dépend du modèle) ;
- sept lignes `Tokens: N ... OK` dans la section `TEST ONNX RUNTIME` ;
- `EXPORT ET TESTS TERMINES`.

Si le script s'arrête avec `ECHEC (code N)`, le code indique l'étape : 2 chargement, 3 remplacement du STFT ou des LSTM, 4 test PyTorch, 5 export, 6 test ONNX Runtime.

## NLLB-200 (traduction)

### Export

```bash
NLLB_OUT_DIR=~/kabyle-models/nllb python export_nllb_onnx.py
```

Le modèle `facebook/nllb-200-distilled-600M` est téléchargé au premier lancement (environ 2,5 Go). Prévoyez environ 5 Go de disque libre : les graphes existent brièvement en 32 bits avant la quantification. La durée va de quelques minutes à quelques dizaines de minutes selon la machine. L'export a été fait avec 15 Go de mémoire vive ; le minimum nécessaire n'a pas été mesuré.

Variables d'environnement :

| Variable | Défaut | Rôle |
|---|---|---|
| `NLLB_MODEL` | `facebook/nllb-200-distilled-600M` | Identifiant Hugging Face ou dossier local |
| `NLLB_OUT_DIR` | `~/kabyle-models/nllb` | Dossier de sortie |
| `NLLB_QUANT` | `1` | `1` quantifie tout en int8, `head` seulement la tête de sortie, `0` rien |
| `NLLB_SRC`, `NLLB_TGT` | `fra_Latn`, `kab_Latn` | Langues source et cible |

Si la qualité de la traduction baisse trop après la quantification complète, relancez avec `NLLB_QUANT=head` : les fichiers sont plus gros, la traduction est plus fidèle. Le script affiche côte à côte la traduction en 32 bits et celle en int8 quand elles diffèrent, ce qui permet d'en juger.

### Résultat attendu

- `tokenizer : BPE`, puis un nombre de fusions BPE (plusieurs centaines de milliers) ;
- `0. identifiants source identiques à tokenizer(texte)` ;
- `1. enveloppes / modèle HF` avec un écart très faible ;
- `3. boucle gloutonne 32 bits / generate() : 6/6 identiques` ;
- `EXPORT TERMINE`.

Le dossier de sortie contient tous les fichiers `nllb_*` de la liste plus haut, ainsi que `nllb_tokens_ref.tsv` (référence pour la vérification du tokenizer, inutile sur le téléphone). La taille totale est de l'ordre de 1 Go : vérifiez avec `ls -lh`.

### Vérifier le tokenizer Java (facultatif, recommandé)

Le tokenizer de l'application est réécrit en Java. Cette commande le compare à celui de Hugging Face sur 27 phrases (JDK 21 requis) :

```bash
mkdir -p build_check
javac -d build_check \
  app/src/main/java/com/kabyleai/app/Tokenizer.java \
  app/src/main/java/com/kabyleai/app/UnigramTokenizer.java \
  app/src/main/java/com/kabyleai/app/BpeTokenizer.java \
  tools/nllb/TokenizerCheck.java
java -cp build_check TokenizerCheck \
  ~/kabyle-models/nllb/nllb_config.txt \
  ~/kabyle-models/nllb/nllb_vocab.tsv \
  ~/kabyle-models/nllb/nllb_tokens_ref.tsv
```

Résultat attendu : `26/27` ou `27/27`. L'écart connu concerne une phrase qui se termine par des espaces : Hugging Face garde un jeton `▁` final, alors que SentencePiece et la classe Java le suppriment. L'application enlève de toute façon les espaces autour de chaque phrase, donc cela n'a pas de conséquence.

Pour tester d'autres phrases, `tools/nllb/tokens_ref.py` écrit une référence à partir d'un fichier de phrases.

### Tester la qualité sans ONNX

`nllb_test.py` traduit une vingtaine de phrases françaises avec PyTorch, pour juger la qualité du kabyle avant tout export :

```bash
python nllb_test.py                       # phrases de test intégrées
python nllb_test.py mes_phrases.txt       # une phrase par ligne
```

## Fadhma-300M (reconnaissance vocale)

Le modèle n'est pas fourni et son script de préparation n'est pas dans le dépôt. L'application attend le fichier `files/fadhma/fadhma_300m_prepared.onnx`. Il prend en entrée 64 000 échantillons audio (4 secondes à 16 kHz, mono) et l'application décode une suite de 40 symboles (lettres kabyles, `[PAD]`, `[UNK]`, `|`, `-`). La source du modèle et sa préparation sont à documenter ici par l'auteur.

## Copier les modèles sur le téléphone

### Téléphone rooté, avec Termux

`build_install.sh` copie les fichiers, corrige le propriétaire (`chown`) et le contexte SELinux (`restorecon`) :

```bash
./build_install.sh ~/kabyle-models/matoub_dynamic ~/kabyle-models/nllb
SKIP_BUILD=1 ./build_install.sh "" ~/kabyle-models/nllb      # NLLB seul, sans recompiler
```

Le script ne copie pas `vocab.json` de Matoub : copiez-le une fois à la main (le dossier `files/matoub/` doit exister) :

```bash
su -c "cp ~/projects/KabyleAI/app/src/main/assets/matoub/vocab.json /data/data/com.kabyleai.app/files/matoub/"
```

Vérifiez aussi que le propriétaire est celui de l'application : `su -c "ls -ln /data/data/com.kabyleai.app/files/matoub"`.

### Téléphone non rooté (ADB et `run-as`)

Cette méthode n'a pas été testée. Elle suppose une version de débogage de l'application (c'est le cas de `assembleDebug`) :

```bash
adb push ~/kabyle-models/nllb /data/local/tmp/nllb
adb shell run-as com.kabyleai.app mkdir -p files/nllb
for f in ~/kabyle-models/nllb/nllb_*; do
  n=$(basename "$f")
  adb shell "run-as com.kabyleai.app sh -c 'cat /data/local/tmp/nllb/$n > files/nllb/$n'"
done
```

Faites de même pour `matoub` et `fadhma`, puis supprimez `/data/local/tmp/nllb` : les fichiers y sont lisibles par tous.

## Vérifier dans l'application

- **Kabyle → Audio** : écrivez `Hemleɣ-k aṭas` et appuyez sur « Lire en kabyle ». Une erreur `Modèle Matoub introuvable` ou `vocab.json introuvable` indique un fichier manquant dans `files/matoub/`.
- **Français → Kabyle** : écrivez une phrase et appuyez sur « Traduire en kabyle ». Le message `Modèle de traduction absent` indique qu'un des fichiers `nllb_*` manque. Le premier appel charge trois modèles et un vocabulaire de plus de 500 000 fusions : il est plus long que les suivants. Le temps de la traduction s'affiche dans l'état.
- **Audio → Kabyle** : appuyez sur « Parler en kabyle », parlez, puis « Arrêter ». `Modele Fadhma introuvable` indique un fichier manquant dans `files/fadhma/`.

## Problèmes fréquents

| Symptôme | Cause probable |
|---|---|
| L'export s'arrête sans message, code 134 | PyTorch du Python de Termux : utilisez l'Ubuntu de `proot-distro`, un PC ou Colab. |
| `Tokenizer ... seuls Unigram et BPE sont gérés` | La version de Transformers fournit un tokenizer d'un autre type. |
| `ATTENTION : seulement N/6 traductions int8 identiques` | La quantification complète dégrade trop : relancez avec `NLLB_QUANT=head`. |
| Erreur `Got invalid dimensions` au premier appel de Matoub | Un ancien modèle à taille fixe est installé : remplacez-le par `matoub_front.onnx` et `matoub_back.onnx`. |
| `No space left on device` pendant l'export | Libérez de la place : les graphes 32 bits sont temporairement gros. |
