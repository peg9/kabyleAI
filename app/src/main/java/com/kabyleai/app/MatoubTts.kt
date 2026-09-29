package com.kabyleai.app

import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

class MatoubTts(private val context: Context) {

    companion object {
        // Deux graphes (voir export_matoub_onnx.py) : le premier prédit les
        // durées, l'application construit l'alignement, le second synthétise.
        private const val FRONT_PATH = "matoub/matoub_front.onnx"
        private const val BACK_PATH = "matoub/matoub_back.onnx"
        private const val VOCAB_PATH = "matoub/vocab.json"

        private const val SAMPLE_RATE = 24000

        // Bornes de l'export dynamique (export_matoub_onnx.py) :
        // au moins les deux "$" de début et de fin, au plus 510 tokens.
        private const val MIN_TOKENS = 2
        private const val MAX_TOKENS = 510

        // Borne de la dimension "frames" de l'export.
        private const val MAX_FRAMES = 20000
    }

    private val environment: OrtEnvironment =
        OrtEnvironment.getEnvironment()

    private var frontSession: OrtSession? = null
    private var backSession: OrtSession? = null
    private var mediaPlayer: MediaPlayer? = null

    private val tokenizer: MatoubTokenizer by lazy {
        MatoubTokenizer(loadVocab())
    }

    private fun loadVocab(): Map<String, Int> {
        val file = File(context.filesDir, VOCAB_PATH)

        if (!file.exists()) {
            throw IllegalStateException(
                "vocab.json introuvable : ${file.absolutePath}"
            )
        }

        val json = JSONObject(file.readText())
        val vocab = mutableMapOf<String, Int>()

        for (key in json.keys()) {
            vocab[key] = json.getInt(key)
        }

        return vocab
    }

    private fun openSession(path: String): OrtSession {
        val model = File(context.filesDir, path)

        if (!model.exists()) {
            throw IllegalStateException(
                "Modèle Matoub introuvable : ${model.absolutePath}"
            )
        }

        val options = OrtSession.SessionOptions()
        options.setIntraOpNumThreads(4)
        options.setInterOpNumThreads(1)

        return environment.createSession(
            model.absolutePath,
            options
        )
    }

    private fun front(): OrtSession {
        if (frontSession == null) {
            frontSession = openSession(FRONT_PATH)
        }
        return frontSession!!
    }

    private fun back(): OrtSession {
        if (backSession == null) {
            backSession = openSession(BACK_PATH)
        }
        return backSession!!
    }

    fun synthesize(
        text: String,
        onSuccess: (File) -> Unit,
        onError: (Exception) -> Unit
    ) {
        Thread {
            try {
                val ids = tokenizer.encode(TtsTextCleaner.clean(text))

                if (ids.size <= MIN_TOKENS) {
                    throw IllegalArgumentException(
                        "Texte vide : rien à synthétiser."
                    )
                }

                if (ids.size > MAX_TOKENS) {
                    throw IllegalArgumentException(
                        "Texte trop long : ${ids.size} tokens, " +
                        "maximum $MAX_TOKENS. Découpez-le en phrases."
                    )
                }

                val frontSess = front()
                val backSess = back()

                OnnxTensor.createTensor(
                    environment,
                    arrayOf(ids)
                ).use { input ->

                    // 1. Durées prédites pour chaque token.
                    frontSess.run(mapOf("input_ids" to input)).use { predicted ->

                        val frames = toLongArray(predicted[1].value)
                        val total = frames.sum().toInt()

                        if (total < 2 || total > MAX_FRAMES) {
                            throw IllegalStateException(
                                "Nombre de trames hors limites : $total"
                            )
                        }

                        // 2. Matrice d'alignement [1, tokens, total].
                        val alignment = buildAlignment(frames, total)

                        OnnxTensor.createTensor(
                            environment,
                            FloatBuffer.wrap(alignment),
                            longArrayOf(1, ids.size.toLong(), total.toLong())
                        ).use { alignmentTensor ->

                            val inputs = mapOf(
                                "input_ids" to input,
                                "d" to (predicted[0] as OnnxTensor),
                                "alignment" to alignmentTensor
                            )

                            // 3. Synthèse de l'audio.
                            backSess.run(inputs).use { result ->

                                val waveform = flatten(result[0].value)

                                if (waveform.isEmpty()) {
                                    throw IllegalStateException(
                                        "Le modèle n'a produit aucun échantillon."
                                    )
                                }

                                if (waveform.any { !it.isFinite() }) {
                                    throw IllegalStateException(
                                        "Le modèle a produit des valeurs " +
                                        "non finies (NaN ou infini)."
                                    )
                                }

                                val wavFile = File(
                                    context.filesDir,
                                    "matoub_output.wav"
                                )

                                writeWav(
                                    wavFile,
                                    waveform,
                                    SAMPLE_RATE
                                )

                                onSuccess(wavFile)
                            }
                        }
                    }
                }

            } catch (e: Exception) {
                onError(e)
            }
        }.start()
    }

    // Ligne t : des 1 sur les trames qui appartiennent au token t.
    private fun buildAlignment(frames: LongArray, total: Int): FloatArray {
        val alignment = FloatArray(frames.size * total)
        var start = 0

        for (token in frames.indices) {
            val end = start + frames[token].toInt()

            for (frame in start until end) {
                alignment[token * total + frame] = 1.0f
            }

            start = end
        }

        return alignment
    }

    private fun toLongArray(value: Any?): LongArray =
        when (value) {
            is LongArray -> value
            is IntArray -> LongArray(value.size) { value[it].toLong() }
            is Array<*> -> toLongArray(value.firstOrNull())
            else -> throw IllegalStateException(
                "Durées ONNX inattendues : " +
                value?.javaClass?.name
            )
        }

    // La sortie "waveform" est [batch, samples] ou [batch, 1, samples]
    // selon l'export ; on garde le premier élément du batch.
    private fun flatten(value: Any?): FloatArray =
        when (value) {
            is FloatArray -> value
            is Array<*> -> flatten(value.firstOrNull())
            else -> throw IllegalStateException(
                "Sortie ONNX inattendue : " +
                value?.javaClass?.name
            )
        }

    fun play(file: File) {
        stop()

        mediaPlayer = MediaPlayer().apply {
            setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(
                        AudioAttributes.CONTENT_TYPE_SPEECH
                    )
                    .build()
            )

            setDataSource(file.absolutePath)

            setOnCompletionListener {
                release()
                mediaPlayer = null
            }

            prepare()
            start()
        }
    }

    fun stop() {
        mediaPlayer?.let {
            try {
                if (it.isPlaying) {
                    it.stop()
                }
            } catch (_: Exception) {
            }

            it.release()
        }

        mediaPlayer = null
    }

    fun close() {
        stop()
        frontSession?.close()
        frontSession = null
        backSession?.close()
        backSession = null
    }

    private fun writeWav(
        file: File,
        samples: FloatArray,
        sampleRate: Int
    ) {
        FileOutputStream(file).use { output ->

            val channels = 1
            val bitsPerSample = 16
            val byteRate =
                sampleRate * channels * bitsPerSample / 8
            val blockAlign =
                channels * bitsPerSample / 8
            val dataSize = samples.size * 2

            output.write("RIFF".toByteArray())
            writeIntLE(output, 36 + dataSize)
            output.write("WAVE".toByteArray())

            output.write("fmt ".toByteArray())
            writeIntLE(output, 16)
            writeShortLE(output, 1)
            writeShortLE(output, channels)
            writeIntLE(output, sampleRate)
            writeIntLE(output, byteRate)
            writeShortLE(output, blockAlign)
            writeShortLE(output, bitsPerSample)

            output.write("data".toByteArray())
            writeIntLE(output, dataSize)

            for (sample in samples) {
                val clipped = sample.coerceIn(-1.0f, 1.0f)

                val pcm = if (clipped < 0) {
                    (clipped * 32768.0f).toInt()
                } else {
                    (clipped * 32767.0f).toInt()
                }

                writeShortLE(output, pcm)
            }
        }
    }

    private fun writeIntLE(
        output: FileOutputStream,
        value: Int
    ) {
        output.write(value and 0xFF)
        output.write((value shr 8) and 0xFF)
        output.write((value shr 16) and 0xFF)
        output.write((value shr 24) and 0xFF)
    }

    private fun writeShortLE(
        output: FileOutputStream,
        value: Int
    ) {
        output.write(value and 0xFF)
        output.write((value shr 8) and 0xFF)
    }
}
