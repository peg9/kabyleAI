from pathlib import Path

p = Path("app/src/main/java/com/footanalytics/app/MainActivity.kt")
s = p.read_text()

needle = "import androidx.compose.material3.Card\n"
insert = """import androidx.compose.material3.Card
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Card"""

if "import androidx.compose.material3.DropdownMenu" not in s:
    s = s.replace(needle, insert, 1)
    p.write_text(s)
    print("Imports dropdown ajoutés")
else:
    print("Imports déjà présents")
