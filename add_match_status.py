from pathlib import Path

p = Path("app/src/main/java/com/footanalytics/app/data/Match.kt")
s = p.read_text()

old = """    val competition: String = "",
    val homeAway: String = "Domicile"
"""

new = """    val competition: String = "",
    val homeAway: String = "Domicile",
    val status: String = "TERMINE"
"""

if old not in s:
    raise SystemExit("Bloc Match introuvable")

s = s.replace(old, new, 1)
p.write_text(s)
print("Champ status ajouté à Match")
