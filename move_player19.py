from pathlib import Path
p = Path("app/src/main/java/com/footanalytics/app/stats/PassNetworkField.kt")
s = p.read_text()
old = """    place(groups["Milieu"].orEmpty(), 0.52f)"""
new = """    place(groups["Milieu"].orEmpty(), 0.52f)
    players.filter { it.number == 19 }.forEach { player ->
        positions[player.id] = FieldPosition(
            x = positions[player.id]?.x ?: 0.5f,
            y = 0.58f
        )
    }"""
if old not in s:
    print("Bloc introuvable")
else:
    s = s.replace(old, new, 1)
    p.write_text(s)
    print("Joueur 19 déplacé à y=0.58")
    print("TAILLE :", p.stat().st_size)
