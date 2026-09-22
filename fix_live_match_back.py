from pathlib import Path

p = Path("app/src/main/java/com/footanalytics/app/MainActivity.kt")
s = p.read_text()

old = """.padding(horizontal = 20.dp)
            .navigationBarsPadding()"""

new = """.padding(top = 52.dp, start = 20.dp, end = 20.dp, bottom = 20.dp)
            .navigationBarsPadding()"""

if old not in s:
    raise SystemExit("Padding LiveMatch introuvable")

s = s.replace(old, new, 1)
p.write_text(s)
print("Padding LiveMatch corrigé")
