#!/data/data/com.termux/files/usr/bin/bash

set -e

PROJECT="/data/data/com.termux/files/home/projects/KabyleAI"
MAIN="$PROJECT/app/src/main/java/com/kabyleai/app/MainActivity.kt"
APK="$PROJECT/app/build/outputs/apk/debug/app-debug.apk"

echo "========================================"
echo " TEST TOKENIZER MATOUB"
echo "========================================"

cd "$PROJECT"

# Sauvegarde
cp "$MAIN" "$MAIN.tokenizer_test.bak"

echo "[1/5] Ajout du test..."

python - <<'PY'
p="/data/data/com.termux/files/home/projects/KabyleAI/app/src/main/java/com/kabyleai/app/MainActivity.kt"

with open(p, encoding="utf-8") as f:
    s=f.read()

test_code = r'''

    private fun testMatoubTokenizer() {
        try {
            val tokenizer = MatoubTokenizer(this)

            val text = "amek i tettiliḍ ayelli s tmurt-iw"

            val phonemes = tokenizer.phonemize(text)
            val ids = tokenizer.encode(text)

            android.util.Log.d("MATOUB_TEST", "TEXT = $text")
            android.util.Log.d("MATOUB_TEST", "PHONEMES = $phonemes")
            android.util.Log.d(
                "MATOUB_TEST",
                "IDS = ${ids.joinToString(",")}"
            )
            android.util.Log.d(
                "MATOUB_TEST",
                "NB IDS = ${ids.size}"
            )

        } catch (e: Exception) {
            android.util.Log.e("MATOUB_TEST", "ERREUR", e)
        }
    }
'''

# Ajouter la fonction avant la dernière accolade de MainActivity
pos = s.rfind("}")
if pos == -1:
    raise SystemExit("Impossible de trouver la fin de MainActivity")

s = s[:pos] + test_code + "\n" + s[pos:]

# Appel automatique dans onCreate
marker = "override fun onCreate(savedInstanceState: Bundle?) {"

pos = s.find(marker)
if pos == -1:
    raise SystemExit("onCreate introuvable")

# Cherche le premier setContent après onCreate
setpos = s.find("setContent {", pos)
if setpos == -1:
    raise SystemExit("setContent introuvable")

# Trouve la fin approximative de setContent en recherchant le prochain
# bloc suivi d'une accolade. On ajoute simplement l'appel au début de onCreate.
insert = s.find("\n", pos) + 1
s = s[:insert] + "        testMatoubTokenizer()\n" + s[insert:]

with open(p, "w", encoding="utf-8") as f:
    f.write(s)

print("Test ajouté")
PY

echo "[2/5] Compilation..."

gradle :app:assembleDebug

echo "[3/5] Installation..."

su -c "pm install -r '$APK'"

echo "[4/5] Nettoyage Logcat..."

su -c "logcat -c"

echo "[5/5] Lancement..."

su -c 'am start -n com.kabyleai.app/.MainActivity' >/dev/null

sleep 3

echo
echo "========================================"
echo " RESULTAT TOKENIZER"
echo "========================================"

su -c 'logcat -d -s MATOUB_TEST:D *:S'

echo
echo "========================================"
echo " RESTAURATION DU FICHIER"
echo "========================================"

mv "$MAIN.tokenizer_test.bak" "$MAIN"

echo "MainActivity.kt restauré."
echo
echo "Test terminé."
