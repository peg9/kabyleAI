from pathlib import Path
p = Path("app/src/main/java/com/footanalytics/app/MainActivity.kt")
s = p.read_text()
lines = s.splitlines(True)
out = []
i = 0
while i < len(lines):
    if i + 1 < len(lines) and lines[i].strip() == "Text(" and lines[i + 1].strip() == ")":
        i += 2
        continue
    out.append(lines[i])
    i += 1
p.write_text("".join(out))
print("Bloc Text vide supprimé")
