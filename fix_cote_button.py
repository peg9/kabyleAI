from pathlib import Path

p = Path("app/src/main/java/com/footanalytics/app/MainActivity.kt")
s = p.read_text()

old = """                    Box {
                        OutlinedTextField(
                            value = positionSide,
                            onValueChange = {},
                            label = { Text("Côté") },
                            readOnly = true,
                            singleLine = true,
                            modifier = Modifier.clickable { sideExpanded = true }
                        )

                        DropdownMenu(
                            expanded = sideExpanded,
                            onDismissRequest = { sideExpanded = false }
                        ) {
                            listOf("Gauche", "Centre", "Droite").forEach { option ->
                                DropdownMenuItem(
                                    text = { Text(option) },
                                    onClick = {
                                        positionSide = option
                                        sideExpanded = false
                                    }
                                )
                            }
                        }
                    }"""

new = """                    Box {
                        Button(onClick = { sideExpanded = true }) {
                            Text("Côté : $positionSide")
                        }

                        DropdownMenu(
                            expanded = sideExpanded,
                            onDismissRequest = { sideExpanded = false }
                        ) {
                            listOf("Gauche", "Centre", "Droite").forEach { option ->
                                DropdownMenuItem(
                                    text = { Text(option) },
                                    onClick = {
                                        positionSide = option
                                        sideExpanded = false
                                    }
                                )
                            }
                        }
                    }"""

if old not in s:
    raise SystemExit("Bloc Côté introuvable")

s = s.replace(old, new, 1)
p.write_text(s)
print("Côté remplacé par un bouton")
