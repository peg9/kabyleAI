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

class MatoubTts(private val context: Context) {

    companion object {
        private const val MODEL_PATH = "matoub/matoub_82m.onnx"
        private const val VOCAB_PATH = "matoub/vocab.json"

        private const val SAMPLE_RATE = 24000
        private const val EXPECTED_TOKENS = 36
        private const val EXPECTED_SAMPLES = 96600
    }

    private val environment: OrtEnvironment =
        OrtEnvironment.getEnvironment()

    private var session: OrtSession? = null
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

    private fun getSession(): OrtSession {
        if (session == null) {
            val model = File(context.filesDir, MODEL_PATH)

            if (!model.exists()) {
                throw IllegalStateException(
                    "Modèle Matoub introuvable : ${model.absolutePath}"
                )
            }

            val options = OrtSession.SessionOptions()
            options.setIntraOpNumThreads(4)
            options.setInterOpNumThreads(1)

            session = environment.createSession(
                model.absolutePath,
                options
            )
        }

        return session!!
    }

    fun synthesize(
        text: String,
        onSuccess: (File) -> Unit,
        onError: (Exception) -> Unit
    ) {
        Thread {
            try {
                val ids = tokenizer.encode(text)

                if (ids.size != EXPECTED_TOKENS) {
                    throw IllegalArgumentException(
                        "Ce modèle ONNX attend exactement " +
                        "$EXPECTED_TOKENS tokens. " +
                        "Le texte produit ${ids.size} tokens."
                    )
                }

                val sess = getSession()

                val input = OnnxTensor.createTensor(
                    environment,
                    arrayOf(ids)
                )

                input.use { tensor ->

                    val inputs = mapOf(
                        "input_ids" to tensor
                    )

                    sess.run(inputs).use { result ->

                        val value = result[0].value

                        val waveform = when (value) {
                            is FloatArray -> value
                            is Array<*> -> {
                                @Suppress("UNCHECKED_CAST")
                                val array = value as Array<FloatArray>
                                array[0]
                            }
                            else -> {
                                throw IllegalStateException(
                                    "Sortie ONNX inattendue : " +
                                    value?.javaClass?.name
                                )
                            }
                        }

                        if (waveform.size != EXPECTED_SAMPLES) {
                            throw IllegalStateException(
                                "Nombre d'échantillons inattendu : " +
                                "${waveform.size}, attendu " +
                                "$EXPECTED_SAMPLES"
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

            } catch (e: Exception) {
                onError(e)
            }
        }.start()
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
        session?.close()
        session = null
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
