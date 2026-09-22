from pathlib import Path
p = Path("app/src/main/java/com/footanalytics/app/MainActivity.kt")
s = p.read_text()
lines = s.splitlines(True)
lines = [x for x in lines if "DEBUG match=" not in x]
p.write_text("".join(lines))
print("DEBUG supprimé")
