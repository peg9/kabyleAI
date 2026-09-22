from pathlib import Path

p = Path("app/src/main/java/com/footanalytics/app/MainActivity.kt")
s = p.read_text()

old = """                Box {
                    OutlinedTextField(
                        value = position,
                        onValueChange = {},
                        label = { Text("Poste") },
                        readOnly = true,
                        singleLine = true,
                        modifier = Modifier.clickable { positionExpanded = true }
                    )

                    DropdownMenu(
                        expanded = positionExpanded,
                        onDismissRequest = { positionExpanded = false }
                    ) {
                        listOf("Gardien", "Défenseur", "Milieu", "Attaquant").forEach { option ->
                            DropdownMenuItem(
                                text = { Text(option) },
                                onClick = {
                                    position = option
                                    positionExpanded = false
                                    if (option == "Gardien") positionSide = "Centre"
                                }
                            )
                        }
                    }
                }"""

new = """                Box {
                    Button(onClick = { positionExpanded = true }) {
                        Text("Poste : $position")
                    }

                    DropdownMenu(
                        expanded = positionExpanded,
                        onDismissRequest = { positionExpanded = false }
                    ) {
                        listOf("Gardien", "Défenseur", "Milieu", "Attaquant").forEach { option ->
                            DropdownMenuItem(
                                text = { Text(option) },
                                onClick = {
                                    position = option
                                    positionExpanded = false
                                    if (option == "Gardien") positionSide = "Centre"
                                }
                            )
                        }
                    }
                }"""

if old not in s:
    raise SystemExit("Bloc Poste introuvable")

s = s.replace(old, new, 1)
p.write_text(s)
print("Poste remplacé par un bouton")
