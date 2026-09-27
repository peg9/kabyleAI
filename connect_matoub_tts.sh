#!/data/data/com.termux/files/usr/bin/bash

set -e

PROJECT="/data/data/com.termux/files/home/projects/KabyleAI"
MAIN="$PROJECT/app/src/main/java/com/kabyleai/app/MainActivity.kt"
BACKUP="$MAIN.before_tts.bak"

cd "$PROJECT"

echo "========================================"
echo " CONNEXION MATOUB TTS"
echo "========================================"

cp "$MAIN" "$BACKUP"

python - <<'PY'
p="app/src/main/java/com/kabyleai/app/MainActivity.kt"

with open(p, encoding="utf-8") as f:
    s=f.read()

# ------------------------------------------------------------
# 1. Signature KabyleAIApp
# ------------------------------------------------------------

old = '''fun KabyleAIApp(
    startRecording: () -> Boolean,
    stopRecording: () -> File?
) {'''

new = '''fun KabyleAIApp(
    startRecording: () -> Boolean,
    stopRecording: () -> File?,
    matoubTts: MatoubTts
) {'''

if "matoubTts: MatoubTts" not in s:

    if old not in s:
        raise SystemExit("ERREUR: signature KabyleAIApp introuvable")

    s = s.replace(old, new, 1)

# ------------------------------------------------------------
# 2. Création MatoubTts dans MainActivity
# ------------------------------------------------------------

if "matoubTts = MatoubTts(this)" not in s:

    marker = '''stopRecording = {
                    stopRecording()
                }'''

    pos = s.find(marker)

    if pos == -1:
        raise SystemExit(
            "ERREUR: appel KabyleAIApp / stopRecording introuvable"
        )

    end = pos + len(marker)

    s = (
        s[:end]
        + ''',
                matoubTts = MatoubTts(this)'''
        + s[end:]
    )

# ------------------------------------------------------------
# 3. Remplacement du placeholder TTS
# ------------------------------------------------------------

placeholder = 'status =\n                            "TTS kabyle : moteur local à connecter"'

if placeholder not in s:

    # Essai avec indentation différente
    import re

    pattern = (
        r'status\s*=\s*'
        r'"TTS kabyle : moteur local à connecter"'
    )

    if not re.search(pattern, s):
        raise SystemExit(
            "ERREUR: placeholder TTS introuvable"
        )

tts_code = '''status = "Synthèse vocale en cours..."

                        matoubTts.synthesize(
                            text = text,

                            onSuccess = { file ->
                                android.os.Handler(
                                    android.os.Looper.getMainLooper()
                                ).post {
                                    status = "Lecture en cours..."
                                    matoubTts.play(file)
                                }
                            },

                            onError = { error ->
                                android.os.Handler(
                                    android.os.Looper.getMainLooper()
                                ).post {
                                    status =
                                        "Erreur TTS : ${error.message}"
                                }
                            }
                        )'''

import re

s = re.sub(
    r'status\s*=\s*"TTS kabyle : moteur local à connecter"',
    tts_code,
    s,
    count=1
)

with open(p, "w", encoding="utf-8") as f:
    f.write(s)

print("Modifications TTS appliquées.")
PY

echo
echo "[1] Vérification..."

grep -n -A15 -B5 "matoubTts" "$MAIN"

echo
echo "[2] Compilation..."

if gradle :app:assembleDebug; then
    echo
    echo "========================================"
    echo " BUILD SUCCESSFUL"
    echo "========================================"
else
    echo
    echo "BUILD FAILED"
    echo "Restauration de MainActivity.kt..."
    cp "$BACKUP" "$MAIN"
    exit 1
fi

echo
echo "APK :"
echo "$PROJECT/app/build/outputs/apk/debug/app-debug.apk"

echo
echo "Sauvegarde :"
echo "$BACKUP"
