from pathlib import Path
p = Path("app/src/main/java/com/footanalytics/app/MainActivity.kt")
s = p.read_text()
lines = s.splitlines(True)
out = []
i = 0
while i < len(lines):
    if lines[i].strip() == "Text(" and i + 2 < len(lines) and "Modifier.padding(8.dp)" in lines[i + 1] and lines[i + 2].strip() == ")":
        i += 3
        continue
    out.append(lines[i])
    i += 1
p.write_text("".join(out))
print("Bloc Text DEBUG résiduel supprimé")
print("TAILLE :", p.stat().st_size)
