from pathlib import Path
p = Path("app/src/main/java/com/footanalytics/app/stats/PassNetworkScreen.kt")
s = p.read_text()
s = s.replace("top = 28.dp", "top = 52.dp", 1)
p.write_text(s)
print("Retour descendu à 52.dp")
print("TAILLE :", p.stat().st_size)
