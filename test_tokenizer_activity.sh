#!/data/data/com.termux/files/usr/bin/bash

set -e

PROJECT="/data/data/com.termux/files/home/projects/KabyleAI"
JAVA_DIR="$PROJECT/app/src/main/java/com/kabyleai/app"
MANIFEST="$PROJECT/app/src/main/AndroidManifest.xml"

TEST_FILE="$JAVA_DIR/TestTokenizerActivity.kt"
BACKUP="$MANIFEST.tokenizer_test.bak"

cd "$PROJECT"

cp "$MANIFEST" "$BACKUP"

cleanup() {
    rm -f "$TEST_FILE"

    if [ -f "$BACKUP" ]; then
        cp "$BACKUP" "$MANIFEST"
        rm -f "$BACKUP"
    fi

    echo
    echo "Nettoyage terminé."
}

trap cleanup EXIT

echo "[1/5] Création du test..."

cat > "$TEST_FILE" <<'KOTLIN'
package com.kabyleai.app

import android.app.Activity
import android.os.Bundle
import android.util.Log
import org.json.JSONObject

class TestTokenizerActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        try {
            val vocabFile =
                java.io.File(filesDir, "matoub/vocab.json")

            if (!vocabFile.exists()) {
                throw Exception(
                    "vocab.json introuvable: ${vocabFile.absolutePath}"
                )
            }

            val json = JSONObject(vocabFile.readText())

            val vocab = mutableMapOf<String, Int>()

            for (key in json.keys()) {
                vocab[key] = json.getInt(key)
            }

            Log.d(
                "MATOUB_TEST",
                "VOCAB SIZE = ${vocab.size}"
            )

            val tokenizer = MatoubTokenizer(vocab)

            val text = "amek i tettiliḍ ayelli s tmurt-iw"

            val phonemes = tokenizer.phonemize(text)
            val ids = tokenizer.encode(text)

            Log.d("MATOUB_TEST", "TEXT = $text")
            Log.d("MATOUB_TEST", "PHONEMES = $phonemes")
            Log.d(
                "MATOUB_TEST",
                "IDS = ${ids.joinToString(",")}"
            )
            Log.d(
                "MATOUB_TEST",
                "NB IDS = ${ids.size}"
            )

        } catch (e: Exception) {
            Log.e("MATOUB_TEST", "ERREUR", e)
        }

        finish()
    }
}
KOTLIN

echo "[2/5] Modification temporaire du Manifest..."

python - <<'PY'
p="app/src/main/AndroidManifest.xml"

with open(p, encoding="utf-8") as f:
    s=f.read()

activity = '''
        <activity
            android:name=".TestTokenizerActivity"
            android:exported="false" />
'''

pos = s.rfind("</application>")

if pos == -1:
    raise SystemExit("Balise </application> introuvable")

s = s[:pos] + activity + s[pos:]

with open(p, "w", encoding="utf-8") as f:
    f.write(s)

print("Manifest OK")
PY

echo "[3/5] Compilation..."

gradle :app:assembleDebug

echo "[4/5] Installation..."

su -c "pm install -r '$PROJECT/app/build/outputs/apk/debug/app-debug.apk'"

echo "[5/5] Exécution..."

su -c 'logcat -c'

su -c 'am start -n com.kabyleai.app/.TestTokenizerActivity' >/dev/null

sleep 2

echo
echo "========================================"
echo " RESULTAT TOKENIZER"
echo "========================================"

su -c 'logcat -d -s MATOUB_TEST:D *:S'

echo
echo "========================================"
echo " FIN"
echo "========================================"
