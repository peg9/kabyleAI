package com.kabyleai.app

import org.json.JSONObject

class MatoubTokenizer(
    private val vocab: Map<String, Int>
) {

    private val vowels = setOf('a', 'e', 'i', 'o', 'u')

    private val backingTriggers =
        setOf('ḍ', 'ṣ', 'ṭ', 'ẓ', 'ṛ', 'q', 'ɣ', 'x')

    private val spirants = mapOf(
        'b' to Pair("β", "b"),
        'd' to Pair("ð", "d"),
        'g' to Pair("ʝ", "ɡ"),
        'k' to Pair("ç", "k"),
        't' to Pair("θ", "t"),
        'ḍ' to Pair("ðˤ", "dˤ")
    )

    private val plain = mapOf(
        'a' to "æ",
        'c' to "ʃ",
        'e' to "ə",
        'f' to "f",
        'h' to "h",
        'i' to "i",
        'j' to "ʒ",
        'l' to "l",
        'm' to "m",
        'n' to "n",
        'o' to "o",
        'p' to "p",
        'q' to "q",
        'r' to "r",
        's' to "s",
        'u' to "u",
        'v' to "v",
        'w' to "w",
        'x' to "χ",
        'y' to "j",
        'z' to "z",
        'č' to "t͡ʃ",
        'ǧ' to "d͡ʒ",
        'ɛ' to "ʕ",
        'ɣ' to "ʁ",
        'ḥ' to "ħ",
        'ṛ' to "rˤ",
        'ṣ' to "sˤ",
        'ṭ' to "tˤ",
        'ẓ' to "zˤ"
    )

    private val nasalAssimilation = mapOf(
        'f' to "m",
        'm' to "m",
        'y' to "ɲ",
        'q' to "ŋ",
        'x' to "ŋ"
    )

    private fun isBacked(chars: List<Char>, position: Int): Boolean {
        val before =
            if (position > 0) chars[position - 1] else null

        val after =
            if (position + 1 < chars.size) chars[position + 1] else null

        return before in backingTriggers ||
                after in backingTriggers
    }

    private fun phonemizeWord(word: String): String {

        val lower = word.lowercase()

        val segments = mutableListOf<Pair<Char, Boolean>>()

        var index = 0

        while (index < lower.length) {

            val char = lower[index]

            val paired =
                index + 1 < lower.length &&
                lower[index + 1] == char &&
                char !in vowels

            segments.add(Pair(char, paired))

            index += if (paired) 2 else 1
        }

        val chars = segments.map { it.first }

        val out = StringBuilder()

        for (position in segments.indices) {

            val char = segments[position].first
            val geminate = segments[position].second

            // Legacy tense t
            if (char == 'ţ') {
                out.append("t")

                if (geminate) {
                    out.append("ː")
                }

                continue
            }

            // Spirants
            val spirant = spirants[char]

            if (spirant != null) {

                out.append(
                    if (geminate) spirant.second
                    else spirant.first
                )

                if (geminate) {
                    out.append("ː")
                }

                continue
            }

            val base = plain[char]
                ?: throw IllegalArgumentException(
                    "Caractère Kabyle non supporté : '$char' (U+" +
                            char.code.toString(16).uppercase() +
                            ")"
                )

            val symbol = when {

                char == 'a' &&
                        isBacked(chars, position) ->
                    "ɑ"

                char == 'n' &&
                        !geminate &&
                        position + 1 < chars.size ->
                    nasalAssimilation[chars[position + 1]]
                        ?: base

                else ->
                    base
            }

            out.append(symbol)

            if (geminate) {
                out.append("ː")
            }
        }

        return out.toString()
    }

    fun phonemize(text: String): String {

        val punctuation = setOf(
            '«', '»', '"', '\'', '“', '”', '‘', '’',
            '.', ',', ';', ':', '!', '?',
            '(', ')', '[', ']', '{', '}', '…'
        )

        val words = text
            .split(Regex("[\\s\\-]+"))
            .map { word ->
                word.filter { char -> char !in punctuation }
            }
            .filter { word -> word.isNotEmpty() }

        return words
            .joinToString(" ") { word ->
                phonemizeWord(word)
            }
            .replace("t͡ʃ", "ʧ")
            .replace("d͡ʒ", "ʤ")
    }

    fun encode(text: String): LongArray {

        val phonemes = phonemize(text)

        val ids = mutableListOf<Long>()

        // $ au début
        ids.add(
            (vocab["$"]
                ?: throw IllegalStateException("Token '$' absent du vocabulaire"))
                .toLong()
        )

        for (symbol in phonemes) {

            val id = vocab[symbol.toString()]
                ?: throw IllegalArgumentException(
                    "Phonème '$symbol' absent du vocabulaire"
                )

            ids.add(id.toLong())
        }

        // $ à la fin
        ids.add(
            (vocab["$"]
                ?: throw IllegalStateException("Token '$' absent du vocabulaire"))
                .toLong()
        )

        return ids.toLongArray()
    }

    companion object {

        fun fromJson(json: String): MatoubTokenizer {

            val obj = JSONObject(json)

            val map = mutableMapOf<String, Int>()

            val keys = obj.keys()

            while (keys.hasNext()) {

                val key = keys.next()

                map[key] = obj.getInt(key)
            }

            return MatoubTokenizer(map)
        }
    }
}
