from pathlib import Path

p = Path("app/src/main/java/com/footanalytics/app/data/FootDatabase.kt")
s = p.read_text()

if "MIGRATION_3_4" not in s:
    marker = "        private val MIGRATION_2_3 = object : Migration(2, 3) {"
    migration = """        private val MIGRATION_3_4 = object : Migration(3, 4) {

            override fun migrate(
                database: SupportSQLiteDatabase
            ) {
                database.execSQL("ALTER TABLE matches ADD COLUMN status TEXT NOT NULL DEFAULT \\TERMINE\")
            }
        }

"""
    if marker not in s:
        raise SystemExit("MIGRATION_2_3 introuvable")
    s = s.replace(marker, migration + marker, 1)

s = s.replace("version = 3", "version = 4", 1)
s = s.replace("                        MIGRATION_2_3\\n                    )", "                        MIGRATION_2_3,\\n                        MIGRATION_3_4\\n                    )", 1)
p.write_text(s)
print("Migration 3->4 ajoutée")
