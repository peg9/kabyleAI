# Compiler et installer l'application

Ce document explique comment produire l'APK de KabyleAI et l'installer sur un téléphone Android.

Seule la compilation sur Termux (directement sur le téléphone) a été exécutée par l'auteur. Les sections Linux et Windows suivent la marche standard d'un projet Android avec Gradle. Elles n'ont pas été testées sur ce dépôt : si une étape échoue, corrigez-la ici.

## Ce qui est compilé

| Élément | Version |
|---|---|
| Android Gradle Plugin | 8.7.3 |
| Kotlin | 2.0.21 |
| JDK | 21 |
| compileSdk / targetSdk | 35 |
| minSdk | 26 (Android 8.0) |
| Jetpack Compose (BOM) | 2025.01.00 |
| ONNX Runtime Android | 1.30.0 |

Le résultat est un APK de débogage : `app/build/outputs/apk/debug/app-debug.apk`. Le paquet s'appelle `com.kabyleai.app` et demande la permission `RECORD_AUDIO`.

## Avant de commencer : deux points propres à ce dépôt

**1. La ligne Termux de `gradle.properties`.** Le fichier contient :

```
android.aapt2FromMavenOverride=/data/data/com.termux/files/usr/bin/aapt2
```

Elle force Gradle à utiliser l'outil `aapt2` installé dans Termux. Sur Linux, Windows ou macOS, ce chemin n'existe pas et la compilation échoue : mettez la ligne en commentaire (`#`) sur ces machines. Ne la commitez pas commentée si vous compilez aussi sur Termux.

**2. Pas de wrapper Gradle.** Le dépôt ne contient ni `gradlew` ni `gradle/wrapper/`. Il faut installer Gradle une fois, puis générer le wrapper :

```
gradle wrapper --gradle-version 8.10.2
```

Le plugin Android 8.7 demande Gradle 8.9 au minimum. Les versions récentes de Gradle fonctionnent en général, mais 8.10.2 est la version que je recommande. Pensez à commiter `gradlew`, `gradlew.bat` et le dossier `gradle/wrapper/` : les autres contributeurs pourront alors compiler avec `./gradlew`.

## Linux

### Avec Android Studio

C'est le chemin le plus simple : Android Studio installe le SDK et gère Gradle.

1. Installez Android Studio (version récente, avec JDK 21 embarqué).
2. Ouvrez le dossier du dépôt, laissez la synchronisation Gradle se terminer.
3. Acceptez d'installer les composants du SDK qu'il demande (plateforme Android 35).
4. Menu *Build > Build APK(s)*, ou dans le terminal du projet : `./gradlew assembleDebug` si le wrapper existe.

### En ligne de commande

Exemple pour Ubuntu ou Debian.

```bash
# JDK 21 (Ubuntu 24.04 ; sinon installez Temurin 21)
sudo apt install openjdk-21-jdk unzip curl git

# SDK Android : téléchargez « Command line tools only » depuis
# https://developer.android.com/studio#command-line-tools-only
mkdir -p ~/android-sdk/cmdline-tools
unzip commandlinetools-linux-*_latest.zip -d ~/android-sdk/cmdline-tools
mv ~/android-sdk/cmdline-tools/cmdline-tools ~/android-sdk/cmdline-tools/latest

export ANDROID_HOME=$HOME/android-sdk
export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools

yes | sdkmanager --licenses
sdkmanager "platform-tools" "platforms;android-35" "build-tools;35.0.0"
```

Ensuite, dans le dossier du dépôt :

```bash
# Ligne aapt2 de Termux : à mettre en commentaire
sed -i 's/^android.aapt2FromMavenOverride/#&/' gradle.properties

# Wrapper Gradle (si absent) : nécessite Gradle installé, voir plus haut
gradle wrapper --gradle-version 8.10.2

./gradlew assembleDebug
```

Si Gradle ne trouve pas le SDK, créez `local.properties` (ignoré par git) :

```
sdk.dir=/home/votre_nom/android-sdk
```

## Windows

### Avec Android Studio

Même marche que sous Linux : installez Android Studio, ouvrez le dossier, laissez Gradle synchroniser, puis *Build > Build APK(s)*. Pensez à mettre en commentaire la ligne `android.aapt2FromMavenOverride` de `gradle.properties`.

### En ligne de commande (PowerShell)

1. Installez un JDK 21, par exemple Temurin :
   ```powershell
   winget install EclipseAdoptium.Temurin.21.JDK
   ```
2. Installez le SDK Android : soit Android Studio, soit *Command line tools only* depuis <https://developer.android.com/studio#command-line-tools-only>, à décompresser dans `%LOCALAPPDATA%\Android\Sdk\cmdline-tools\latest`.
3. Installez Gradle 8.10.2 depuis <https://gradle.org/releases/> et ajoutez son dossier `bin` au `PATH`.
4. Dans PowerShell, à la racine du dépôt :

```powershell
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
& "$env:ANDROID_HOME\cmdline-tools\latest\bin\sdkmanager.bat" --licenses
& "$env:ANDROID_HOME\cmdline-tools\latest\bin\sdkmanager.bat" "platform-tools" "platforms;android-35" "build-tools;35.0.0"

# Ligne aapt2 de Termux : à mettre en commentaire
(Get-Content gradle.properties) -replace '^android.aapt2FromMavenOverride','#android.aapt2FromMavenOverride' | Set-Content gradle.properties

gradle wrapper --gradle-version 8.10.2
.\gradlew.bat assembleDebug
```

Si Gradle ne trouve pas le SDK, créez `local.properties` à la racine. Dans ce fichier, les deux-points et les antislashs du chemin s'écrivent avec un antislash devant :

```
sdk.dir=C\:\\Users\\VotreNom\\AppData\\Local\\Android\\Sdk
```

## Android (Termux, directement sur le téléphone)

C'est la méthode que l'auteur utilise. Elle demande un téléphone Android avec [Termux](https://termux.dev) (version F-Droid) et de la place : comptez plusieurs Go pour l'outillage de compilation.

Ce qui est établi par l'usage :

- `gradle :app:assembleDebug` compile l'application dans Termux.
- `gradle.properties` pointe vers `aapt2` de Termux, `/data/data/com.termux/files/usr/bin/aapt2` : c'est le rôle de la ligne décrite plus haut.
- Le script `build_install.sh` compile, installe l'APK avec `su -c "pm install -r ..."` (téléphone rooté), copie les modèles et lance l'application.

À compléter par l'auteur : les paquets Termux et le SDK Android exacts utilisés (JDK 21, Gradle, `aapt2`, plateforme Android 35, emplacement du SDK). Cette information n'est pas dans le dépôt, et je ne l'ai pas reconstituée pour ne pas écrire une procédure inexacte.

Utilisation du script :

```bash
cd ~/projects/KabyleAI
./build_install.sh                          # compile et installe seulement
./build_install.sh <dossier_matoub> <dossier_nllb>   # avec les modèles
SKIP_BUILD=1 ./build_install.sh "" <dossier_nllb>    # copie les modèles sans recompiler
```

Sans root, le script ouvre l'installateur Android avec `termux-open` et s'arrête : les modèles doivent alors être copiés autrement (voir [MODELES.md](MODELES.md)).

## Installer l'APK sur un téléphone

### Depuis un ordinateur (ADB)

Sur le téléphone : activez les *Options pour les développeurs* (appui répété sur le numéro de build), puis le *Débogage USB*. Branchez-le, puis :

```bash
adb devices                # le téléphone doit apparaître, après autorisation à l'écran
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Sous Windows : `& "$env:ANDROID_HOME\platform-tools\adb.exe" install -r app\build\outputs\apk\debug\app-debug.apk`.

### Sans ordinateur

Copiez `app-debug.apk` sur le téléphone, ouvrez-le depuis un gestionnaire de fichiers et autorisez l'installation depuis cette source.

L'application vide n'affiche que des messages d'erreur tant que les modèles ne sont pas installés : passez ensuite à [MODELES.md](MODELES.md).

## Dépannage

| Message | Cause probable |
|---|---|
| `SDK location not found` | Créez `local.properties` avec `sdk.dir`, ou définissez `ANDROID_HOME`. |
| Erreur sur `aapt2` pendant `processDebugResources` | Typiquement, la ligne Termux de `gradle.properties` est encore active sur un PC. |
| `Unsupported class file major version 65` ou `invalid source release: 21` | Gradle tourne avec un JDK plus ancien que 21. Vérifiez `java -version` et `JAVA_HOME`. |
| `Minimum supported Gradle version is 8.9` | Gradle trop ancien : utilisez 8.10.2. |
| `Could not resolve com.android.tools.build:gradle` ou `Received status code 429/403` | Problème réseau ou limite de débit des dépôts Maven et Google. Réessayez plus tard, vérifiez le proxy. |
| `OutOfMemoryError: Java heap space` | Augmentez `-Xmx2048m` dans `org.gradle.jvmargs` de `gradle.properties`. |
| Installation refusée : `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | Une autre version signée autrement est installée : désinstallez-la d'abord. |
