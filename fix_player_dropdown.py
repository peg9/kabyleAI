from pathlib import Path

p = Path("app/src/main/java/com/footanalytics/app/MainActivity.kt")
s = p.read_text()

s = s.replace("import androidx.compose.foundation.border\n", "import androidx.compose.foundation.border\nimport androidx.compose.foundation.clickable\nimport androidx.compose.foundation.layout.Box\n", 1)

old = """                Column {
                    OutlinedTextField(
                        value = position,
                        onValueChange = {},
                        label = { Text("Poste") },
                        readOnly = true,
                        singleLine = true
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

                    TextButton(onClick = { positionExpanded = true }) {
                        Text("Choisir le poste")
                    }
                }"""

new = """                Box {
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

if old not in s:
    raise SystemExit("Bloc Poste introuvable")
s = s.replace(old, new, 1)

old = """                    Column {
                        OutlinedTextField(
                            value = positionSide,
                            onValueChange = {},
                            label = { Text("Côté") },
                            readOnly = true,
                            singleLine = true
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

                        TextButton(onClick = { sideExpanded = true }) {
                            Text("Choisir le côté")
                        }
                    }"""

new = """                    Box {
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

if old not in s:
    raise SystemExit("Bloc Côté introuvable")
s = s.replace(old, new, 1)

p.write_text(s)
print("Dropdown Poste/Côté corrigé")
