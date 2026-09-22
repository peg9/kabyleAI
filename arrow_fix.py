from pathlib import Path
p=Path("app/src/main/java/com/footanalytics/app/stats/PassNetworkField.kt")
s=p.read_text()
a=s.index("            // Connexions de passes")
b=s.index("            // Joueurs : uniquement le numéro",a)
new="""            // Connexions de passes : flèches
            passNetwork.forEach { entry ->
                val passer = players.firstOrNull { it.name == entry.passerName }
                val receiver = players.firstOrNull { it.name == entry.receiverName }
                val startPosition = passer?.let { positions[it.id] }
                val endPosition = receiver?.let { positions[it.id] }

                if (startPosition != null && endPosition != null) {
                    val start = Offset(startPosition.x * w, startPosition.y * h)
                    val end = Offset(endPosition.x * w, endPosition.y * h)
                    val dx = end.x - start.x
                    val dy = end.y - start.y
                    val distance = kotlin.math.sqrt(dx * dx + dy * dy)

                    if (distance > 1f) {
                        val stroke = max(2f, entry.count * 1.5f).coerceAtMost(12f).dp.toPx()
                        val angle = kotlin.math.atan2(dy, dx)
                        val radius = 20.dp.toPx()

                        val arrowEnd = Offset(
                            end.x - kotlin.math.cos(angle) * radius,
                            end.y - kotlin.math.sin(angle) * radius
                        )

                        drawLine(
                            color = Color.White.copy(alpha = 0.75f),
                            start = start,
                            end = arrowEnd,
                            strokeWidth = stroke
                        )

                        val length = max(8.dp.toPx(), stroke * 2.5f)
                        val spread = 0.55f

                        val left = Offset(
                            arrowEnd.x - kotlin.math.cos(angle - spread) * length,
                            arrowEnd.y - kotlin.math.sin(angle - spread) * length
                        )

                        val right = Offset(
                            arrowEnd.x - kotlin.math.cos(angle + spread) * length,
                            arrowEnd.y - kotlin.math.sin(angle + spread) * length
                        )

                        drawLine(
                            color = Color.White.copy(alpha = 0.75f),
                            start = arrowEnd,
                            end = left,
                            strokeWidth = stroke
                        )

                        drawLine(
                            color = Color.White.copy(alpha = 0.75f),
                            start = arrowEnd,
                            end = right,
                            strokeWidth = stroke
                        )
                    }
                }
            }

"""
p.write_text(s[:a]+new+s[b:])
print("Flèches ajoutées")
