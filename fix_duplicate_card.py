from pathlib import Path

p = Path("app/src/main/java/com/footanalytics/app/MainActivity.kt")
s = p.read_text()
s = s.replace("import androidx.compose.material3.DropdownMenuItem\nimport androidx.compose.material3.Card\n", "import androidx.compose.material3.DropdownMenuItem\n", 1)
p.write_text(s)
print("Doublon Card supprimé")
