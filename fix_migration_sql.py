from pathlib import Path

p = Path("app/src/main/java/com/footanalytics/app/data/FootDatabase.kt")
s = p.read_text()
lines = s.splitlines()
for i, line in enumerate(lines):
    if "ALTER TABLE matches ADD COLUMN status" in line:
        lines[i] = '                database.execSQL("ALTER TABLE matches ADD COLUMN status TEXT NOT NULL DEFAULT '"'TERMINE'"'")'
p.write_text("\\n".join(lines) + "\\n")
print("Ligne SQL remplacée")
