from pathlib import Path

p = Path("app/src/main/java/com/footanalytics/app/MainActivity.kt")
s = p.read_text()
s = s.replace("import androidx.compose.material3.Cardimport androidx.compose.material3.MaterialTheme", "import androidx.compose.material3.Card\nimport androidx.compose.material3.MaterialTheme")
p.write_text(s)
print("Import corrigé")
