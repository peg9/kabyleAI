from pathlib import Path

p = Path("app/src/main/java/com/footanalytics/app/MainActivity.kt")
s = p.read_text()

old = """    var position by remember { mutableStateOf("Milieu") }
    var secondary by remember { mutableStateOf("") }
    var foot by remember { mutableStateOf("Droit") }"""

new = """    var position by remember { mutableStateOf("Milieu") }
    var positionSide by remember { mutableStateOf("Centre") }
    var positionExpanded by remember { mutableStateOf(false) }
    var sideExpanded by remember { mutableStateOf(false) }
    var secondary by remember { mutableStateOf("") }
    var foot by remember { mutableStateOf("Droit") }"""

if old not in s:
    raise SystemExit("Bloc des variables introuvable")

s = s.replace(old, new, 1)

old = """                OutlinedTextField(
                    value = position,
                    onValueChange = { position = it },
                    label = { Text("Poste") },
                    singleLine = true
                )"""

new = """                Column {
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
                }

                if (position != "Gardien") {
                    Column {
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
                    }
                }"""

if old not in s:
    raise SystemExit("Champ Poste introuvable")

s = s.replace(old, new, 1)

old = """                        position.trim(),
                        secondary.trim(),"""

new = """                        if (position == "Gardien") position else "$position $positionSide",
                        secondary.trim(),"""

if old not in s:
    raise SystemExit("Appel onCreate introuvable")

s = s.replace(old, new, 1)

p.write_text(s)
print("Sélecteur de poste installé")
