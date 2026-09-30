package com.kabyleai.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import java.io.File
import java.io.RandomAccessFile
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread
import kotlin.math.sqrt

private const val SAMPLE_RATE = 16000
private const val CHANNELS = 1
private const val BITS_PER_SAMPLE = 16

private const val FADHMA_SAMPLES = 64000
private const val FADHMA_MODEL = "fadhma/fadhma_300m_prepared.onnx"

private val FADHMA_VOCAB = arrayOf(
    "[PAD]", "[UNK]", "|", "-",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z",
    "č", "ǧ", "ɛ", "ɣ", "ḍ", "ḥ", "ṛ", "ṣ", "ṭ", "ẓ"
)

class MainActivity : ComponentActivity() {

    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null
    private var audioFile: File? = null
    private var isRecording = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        BundledModelFiles.ensureMatoubVocab(this)

        try {
            ortEnvironment = OrtEnvironment.getEnvironment()
        } catch (e: Exception) {
            e.printStackTrace()
        }

        setContent {
            KabyleAIApp(
                startRecording = {
                    startRecording()
                },
                stopRecording = {
                    stopRecording()
                },
                matoubTts = MatoubTts(this),
                translator = NllbTranslator(File(filesDir, "nllb"))
            )
        }
    }


    private fun loadWavForFadhma(file: File): FloatArray {
        val samples = FloatArray(FADHMA_SAMPLES)

        java.io.FileInputStream(file).use { input ->
            val header = ByteArray(44)
            var headerRead = 0

            while (headerRead < 44) {
                val n = input.read(header, headerRead, 44 - headerRead)
                if (n < 0) break
                headerRead += n
            }

            require(headerRead == 44) { "WAV invalide : en-tete incomplet" }

            val channels = (header[22].toInt() and 0xff) or
                    ((header[23].toInt() and 0xff) shl 8)

            val sampleRate = (header[24].toInt() and 0xff) or
                    ((header[25].toInt() and 0xff) shl 8) or
                    ((header[26].toInt() and 0xff) shl 16) or
                    ((header[27].toInt() and 0xff) shl 24)

            val bitsPerSample = (header[34].toInt() and 0xff) or
                    ((header[35].toInt() and 0xff) shl 8)

            require(channels == 1) {
                "Fadhma attend un WAV mono"
            }

            require(sampleRate == 16000) {
                "Frequence WAV invalide : ${sampleRate} Hz"
            }

            require(bitsPerSample == 16) {
                "Fadhma attend du PCM16"
            }

            val pcm = ByteArray(4096)
            val raw = ArrayList<Float>()

            while (true) {
                val n = input.read(pcm)
                if (n <= 0) break

                var i = 0
                while (i + 1 < n) {
                    val lo = pcm[i].toInt() and 0xff
                    val hi = pcm[i + 1].toInt()
                    val value = (hi shl 8) or lo

                    raw.add(value / 32768.0f)
                    i += 2
                }
            }

            val count = minOf(raw.size, FADHMA_SAMPLES)

            for (i in 0 until count) {
                samples[i] = raw[i]
            }

            if (count > 0) {
                var mean = 0.0f
                for (i in 0 until count) {
                    mean += samples[i]
                }
                mean /= count

                var variance = 0.0f
                for (i in 0 until count) {
                    val d = samples[i] - mean
                    variance += d * d
                }

                val std = kotlin.math.sqrt(
                    variance / count
                ).coerceAtLeast(1e-7f)

                for (i in 0 until count) {
                    samples[i] = (samples[i] - mean) / std
                }
            }
        }

        return samples
    }


    fun transcribeFadhma(
        wavFile: File,
        onResult: (String) -> Unit,
        onError: (String) -> Unit
    ) {
        Thread {
            try {
                val env = ortEnvironment
                    ?: throw IllegalStateException("ONNX Runtime non initialise")

                val modelFile = File(filesDir, FADHMA_MODEL)

                if (!modelFile.exists()) {
                    throw IllegalStateException(
                        "Modele Fadhma introuvable : ${modelFile.absolutePath}"
                    )
                }

                if (ortSession == null) {
                    val options = OrtSession.SessionOptions()
                    options.setIntraOpNumThreads(4)
                    options.setInterOpNumThreads(1)

                    ortSession = env.createSession(
                        modelFile.absolutePath,
                        options
                    )

                    options.close()
                }

                val samples = loadWavForFadhma(wavFile)

                val input = arrayOf(samples)

                OnnxTensor.createTensor(env, input).use { tensor ->

                    val session = ortSession
                        ?: throw IllegalStateException("Session ONNX indisponible")

                    session.run(
                        mapOf("input_values" to tensor)
                    ).use { result ->

                        val value = result[0].value

                        val batch = value as Array<*>
                        val frames = batch[0] as Array<*>

                        val text = StringBuilder()

                        var previous = -1

                        for (frame in frames) {
                            val logits = frame as FloatArray

                            var best = 0
                            var bestValue = logits[0]

                            for (i in 1 until logits.size) {
                                if (logits[i] > bestValue) {
                                    bestValue = logits[i]
                                    best = i
                                }
                            }

                            // CTC : supprimer les repetitions
                            // et le token blank = 0
                            if (best != 0 && best != previous) {
                                if (best < FADHMA_VOCAB.size) {
                                    val token = FADHMA_VOCAB[best]

                                    if (token == "|") {
                                        text.append(' ')
                                    } else {
                                        text.append(token)
                                    }
                                }
                            }

                            previous = best
                        }

                        val transcription = text.toString()
                            .trim()
                            .replace(Regex("\\s+"), " ")

                        runOnUiThread {
                            onResult(transcription)
                        }
                    }
                }

            } catch (e: Exception) {
                e.printStackTrace()

                runOnUiThread {
                    onError(
                        e.message ?: e.javaClass.simpleName
                    )
                }
            }
        }.start()
    }

    private fun startRecording(): Boolean {

        if (isRecording) {
            return false
        }

        val minBuffer = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        if (minBuffer <= 0) {
            return false
        }

        val bufferSize = maxOf(
            minBuffer,
            SAMPLE_RATE * 2
        )

        val file = File(
            filesDir,
            "kabyle_input.wav"
        )

        audioFile = file

        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )

        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            recorder.release()
            return false
        }

        audioRecord = recorder
        isRecording = true

        recordingThread = thread(
            start = true,
            name = "KabyleAudioRecord"
        ) {

            try {

                RandomAccessFile(
                    file,
                    "rw"
                ).use { wav ->

                    writeWavHeader(
                        wav,
                        0
                    )

                    val buffer = ShortArray(bufferSize / 2)

                    recorder.startRecording()

                    while (isRecording) {

                        val count = recorder.read(
                            buffer,
                            0,
                            buffer.size
                        )

                        if (count > 0) {

                            val bytes = ByteArray(
                                count * 2
                            )

                            for (i in 0 until count) {

                                val sample = buffer[i].toInt()

                                bytes[i * 2] =
                                    (sample and 0xff).toByte()

                                bytes[i * 2 + 1] =
                                    ((sample shr 8) and 0xff).toByte()
                            }

                            wav.write(bytes)
                        }
                    }

                    val dataSize =
                        wav.length() - 44L

                    writeWavHeader(
                        wav,
                        dataSize
                    )
                }

            } catch (_: Exception) {

                // L'état sera remonté par stopRecording().
            }
        }

        return true
    }

    private fun stopRecording(): File? {

        if (!isRecording) {
            return audioFile
        }

        isRecording = false

        recordingThread?.join(1000)
        recordingThread = null

        audioRecord?.runCatching {
            stop()
            release()
        }

        audioRecord = null

        return audioFile
    }

    private fun writeWavHeader(
        file: RandomAccessFile,
        dataSize: Long
    ) {

        file.seek(0)

        val byteRate =
            SAMPLE_RATE *
                CHANNELS *
                BITS_PER_SAMPLE / 8

        val blockAlign =
            CHANNELS *
                BITS_PER_SAMPLE / 8

        file.writeBytes("RIFF")
        writeIntLE(
            file,
            (36 + dataSize).toInt()
        )

        file.writeBytes("WAVE")

        file.writeBytes("fmt ")
        writeIntLE(file, 16)
        writeShortLE(file, 1)
        writeShortLE(file, CHANNELS)
        writeIntLE(file, SAMPLE_RATE)
        writeIntLE(file, byteRate)
        writeShortLE(file, blockAlign)
        writeShortLE(file, BITS_PER_SAMPLE)

        file.writeBytes("data")
        writeIntLE(
            file,
            dataSize.toInt()
        )
    }

    private fun writeIntLE(
        file: RandomAccessFile,
        value: Int
    ) {
        file.write(value and 0xff)
        file.write((value shr 8) and 0xff)
        file.write((value shr 16) and 0xff)
        file.write((value shr 24) and 0xff)
    }

    private fun writeShortLE(
        file: RandomAccessFile,
        value: Int
    ) {
        file.write(value and 0xff)
        file.write((value shr 8) and 0xff)
    }

    override fun onDestroy() {

        stopRecording()

        ortSession?.close()
        ortEnvironment?.close()

        ortSession = null
        ortEnvironment = null

        super.onDestroy()
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun KabyleAIApp(
    startRecording: () -> Boolean,
    stopRecording: () -> File?,
    matoubTts: MatoubTts,
    translator: NllbTranslator
) {

    val context = LocalContext.current
    val clipboard = LocalClipboardManager.current
    val mainHandler = remember {
        android.os.Handler(android.os.Looper.getMainLooper())
    }

    var french by remember {
        mutableStateOf("")
    }

    var translation by remember {
        mutableStateOf("")
    }

    var translating by remember {
        mutableStateOf(false)
    }

    var modelOutput by remember {
        mutableStateOf("")
    }

    val correctionsFile = remember {
        File(context.filesDir, "corrections.tsv")
    }

    var correctionCount by remember {
        mutableStateOf(CorrectionStore.count(correctionsFile))
    }

    var text by remember {
        mutableStateOf("Hemleɣ-k aṭas")
    }

    var transcription by remember {
        mutableStateOf("")
    }

    var recording by remember {
        mutableStateOf(false)
    }

    var status by remember {
        mutableStateOf("Prêt")
    }

    var importing by remember {
        mutableStateOf(false)
    }

    var importFraction by remember {
        mutableStateOf(0f)
    }

    var importText by remember {
        mutableStateOf("")
    }

    var modelStatus by remember {
        mutableStateOf(ModelCatalog.summary(context.filesDir))
    }

    val cancelImport = remember {
        AtomicBoolean(false)
    }

    val startImport: (ModelSource) -> Unit = { source ->

        if (!importing) {

            importing = true
            importFraction = 0f
            importText = "Préparation..."
            cancelImport.set(false)

            (context as? Activity)?.window?.addFlags(
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
            )

            thread {

                val report = ModelImporter.importFrom(
                    source,
                    context.filesDir,
                    ModelImporter.ProgressListener { fraction, message ->
                        mainHandler.post {
                            importFraction = fraction
                            importText = message
                        }
                    },
                    cancelImport
                )

                BundledModelFiles.ensureMatoubVocab(context)

                mainHandler.post {
                    importing = false
                    modelStatus = ModelCatalog.summary(context.filesDir)
                    status = report.message

                    (context as? Activity)?.window?.clearFlags(
                        WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
                    )
                }
            }
        }
    }

    val exportLauncher =
        rememberLauncherForActivityResult(
            ActivityResultContracts.CreateDocument("text/tab-separated-values")
        ) { uri: Uri? ->

            if (uri != null) {
                thread {
                    val message = try {
                        val stream = context.contentResolver.openOutputStream(uri)
                            ?: throw java.io.IOException("destination illisible")
                        val n = stream.use { CorrectionStore.export(correctionsFile, it) }
                        "$n correction(s) exportée(s)"
                    } catch (e: Exception) {
                        "Export impossible : ${e.message}"
                    }
                    mainHandler.post { status = message }
                }
            }
        }

    val folderLauncher =
        rememberLauncherForActivityResult(
            ActivityResultContracts.OpenDocumentTree()
        ) { uri: Uri? ->

            if (uri != null) {
                startImport(SafModelSource.ofFolder(context, uri))
            }
        }

    val filesLauncher =
        rememberLauncherForActivityResult(
            ActivityResultContracts.OpenMultipleDocuments()
        ) { uris: List<Uri> ->

            if (uris.isNotEmpty()) {
                startImport(SafModelSource.ofFiles(context, uris))
            }
        }

    val permissionLauncher =
        rememberLauncherForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->

            status =
                if (granted) {
                    "Microphone autorisé"
                } else {
                    "Permission microphone refusée"
                }
        }

    MaterialTheme {

        Scaffold(

            topBar = {

                TopAppBar(
                    title = {
                        Text("Kabyle AI")
                    }
                )
            }

        ) { padding ->

            Column(

                modifier = Modifier
                    .padding(padding)
                    .padding(20.dp)
                    .fillMaxSize()
                    .verticalScroll(
                        rememberScrollState()
                    ),

                verticalArrangement =
                    Arrangement.spacedBy(14.dp)
            ) {

                Text(
                    text = "Français → Kabyle",
                    style =
                        MaterialTheme.typography.titleLarge
                )

                OutlinedTextField(
                    value = french,

                    onValueChange = {
                        french = it
                    },

                    modifier =
                        Modifier.fillMaxWidth(),

                    minLines = 3,

                    label = {
                        Text("Texte français")
                    }
                )

                Button(

                    enabled = !translating,

                    onClick = {

                        val input = french

                        if (input.isBlank()) {

                            status = "Écrivez d'abord un texte français"

                        } else if (!translator.isInstalled()) {

                            status =
                                "Modèle de traduction absent : utilisez « Importer les modèles » en bas de l'écran"

                        } else {

                            translating = true
                            status = "Traduction en cours..."

                            thread {

                                val started = System.currentTimeMillis()

                                try {

                                    val result = translator.translate(input)
                                    val seconds =
                                        (System.currentTimeMillis() - started) / 1000

                                    mainHandler.post {
                                        translation = result
                                        modelOutput = result
                                        translating = false
                                        status =
                                            "Traduction terminée ($seconds s)"
                                    }

                                } catch (e: Throwable) {

                                    mainHandler.post {
                                        translating = false
                                        status =
                                            "Erreur traduction : ${e.message}"
                                    }
                                }
                            }
                        }
                    },

                    modifier =
                        Modifier.fillMaxWidth()
                ) {

                    Text(
                        if (translating) {
                            "Traduction en cours..."
                        } else {
                            "Traduire en kabyle"
                        }
                    )
                }

                OutlinedTextField(
                    value = translation,

                    onValueChange = {
                        translation = it
                    },

                    modifier =
                        Modifier.fillMaxWidth(),

                    minLines = 3,

                    label = {
                        Text("Traduction kabyle (à relire)")
                    }
                )

                Row(
                    horizontalArrangement =
                        Arrangement.spacedBy(10.dp),
                    modifier =
                        Modifier.fillMaxWidth()
                ) {

                    Button(

                        onClick = {

                            if (translation.isBlank()) {

                                status = "Rien à lire : traduisez d'abord"

                            } else {

                                status = "Synthèse vocale en cours..."

                                matoubTts.synthesize(
                                    text = translation,

                                    onSuccess = { file ->
                                        mainHandler.post {
                                            status = "Lecture en cours..."
                                            matoubTts.play(file)
                                        }
                                    },

                                    onError = { error ->
                                        mainHandler.post {
                                            status =
                                                "Erreur TTS : ${error.message}"
                                        }
                                    }
                                )
                            }
                        },

                        modifier =
                            Modifier.weight(1f)
                    ) {

                        Text("Lire")
                    }

                    OutlinedButton(

                        onClick = {

                            if (translation.isNotBlank()) {
                                clipboard.setText(
                                    AnnotatedString(translation)
                                )
                                status = "Traduction copiée"
                            }
                        },

                        modifier =
                            Modifier.weight(1f)
                    ) {

                        Text("Copier")
                    }
                }

                Row(
                    horizontalArrangement =
                        Arrangement.spacedBy(10.dp),
                    modifier =
                        Modifier.fillMaxWidth()
                ) {

                    OutlinedButton(

                        onClick = {

                            if (french.isBlank() || translation.isBlank()) {

                                status = "Traduisez et corrigez d'abord"

                            } else {

                                val fr = french
                                val original = modelOutput
                                val corrected = translation

                                thread {

                                    val message = try {
                                        when (CorrectionStore.save(
                                            correctionsFile, fr, original, corrected
                                        )) {
                                            CorrectionStore.Result.ADDED ->
                                                "Correction enregistrée"
                                            CorrectionStore.Result.UPDATED ->
                                                "Correction mise à jour"
                                            CorrectionStore.Result.UNCHANGED ->
                                                "Déjà enregistrée"
                                            else ->
                                                "Rien à enregistrer"
                                        }
                                    } catch (e: Exception) {
                                        "Enregistrement impossible : ${e.message}"
                                    }

                                    mainHandler.post {
                                        correctionCount =
                                            CorrectionStore.count(correctionsFile)
                                        status = message
                                    }
                                }
                            }
                        },

                        modifier =
                            Modifier.weight(1f)
                    ) {

                        Text("Enregistrer la correction")
                    }

                    OutlinedButton(

                        enabled = correctionCount > 0,

                        onClick = {
                            exportLauncher.launch("corrections_kabyle.tsv")
                        },

                        modifier =
                            Modifier.weight(1f)
                    ) {

                        Text("Exporter ($correctionCount)")
                    }
                }

                Text(
                    text =
                        "Traduction automatique (NLLB-200) : la qualité " +
                        "du kabyle est inégale, à relire avant usage.",

                    style =
                        MaterialTheme.typography.bodySmall
                )

                HorizontalDivider()

                Text(
                    text = "Kabyle → Audio",
                    style =
                        MaterialTheme.typography.titleLarge
                )

                OutlinedTextField(
                    value = text,

                    onValueChange = {
                        text = it
                    },

                    modifier =
                        Modifier.fillMaxWidth(),

                    minLines = 5,

                    label = {
                        Text("Texte kabyle")
                    }
                )

                Button(

                    onClick = {

                        status = "Synthèse vocale en cours..."

                        matoubTts.synthesize(
                            text = text,

                            onSuccess = { file ->
                                android.os.Handler(
                                    android.os.Looper.getMainLooper()
                                ).post {
                                    status = "Lecture en cours..."
                                    matoubTts.play(file)
                                }
                            },

                            onError = { error ->
                                android.os.Handler(
                                    android.os.Looper.getMainLooper()
                                ).post {
                                    status =
                                        "Erreur TTS : ${error.message}"
                                }
                            }
                        )
                    },

                    modifier =
                        Modifier.fillMaxWidth()
                ) {

                    Text("Lire en kabyle")
                }

                HorizontalDivider()

                Text(
                    text = "Audio → Kabyle",
                    style =
                        MaterialTheme.typography.titleLarge
                )

                Button(

                    onClick = {

                        val permission =
                            ContextCompat.checkSelfPermission(
                                context,
                                Manifest.permission.RECORD_AUDIO
                            )

                        if (
                            permission !=
                            PackageManager.PERMISSION_GRANTED
                        ) {

                            permissionLauncher.launch(
                                Manifest.permission.RECORD_AUDIO
                            )

                        } else {

                            if (!recording) {

                                val started =
                                    startRecording()

                                if (started) {

                                    recording = true

                                    status =
                                        "Enregistrement WAV 16 kHz..."

                                } else {

                                    status =
                                        "Impossible de démarrer le microphone"
                                }

                            } else {

                                val file =
                                    stopRecording()

                                recording = false

                                if (file != null && file.exists()) {

                                    status =
                                        "Transcription Fadhma en cours..."

                                    transcription = ""

                                    (context as? MainActivity)?.transcribeFadhma(
                                        file,
                                        onResult = { result ->
                                            transcription = result

                                            if (result.isNotBlank()) {
                                                status = "Synthèse vocale en cours..."

                                                matoubTts.synthesize(
                                                    text = result,
                                                    onSuccess = { file ->
                                                        android.os.Handler(
                                                            android.os.Looper.getMainLooper()
                                                        ).post {
                                                            status = "Lecture en cours..."
                                                            matoubTts.play(file)
                                                        }
                                                    },
                                                    onError = { error ->
                                                        android.os.Handler(
                                                            android.os.Looper.getMainLooper()
                                                        ).post {
                                                            status =
                                                                "Erreur TTS : ${error.message}"
                                                        }
                                                    }
                                                )
                                            } else {
                                                status = "Aucun texte reconnu"
                                            }
                                        },
                                        onError = { error ->
                                            status =
                                                "Erreur Fadhma : $error"
                                        }
                                    )

                                } else {

                                    status =
                                        "Erreur lors de l'enregistrement"
                                }
                            }
                        }
                    },

                    modifier =
                        Modifier.fillMaxWidth()
                ) {

                    Text(

                        if (recording) {
                            "Arrêter"
                        } else {
                            "Parler en kabyle"
                        }
                    )
                }

                OutlinedTextField(

                    value = transcription,

                    onValueChange = {
                        transcription = it
                    },

                    modifier =
                        Modifier.fillMaxWidth(),

                    minLines = 5,

                    label = {
                        Text("Texte reconnu")
                    }
                )

                HorizontalDivider()

                Text(
                    text = "Modèles",
                    style =
                        MaterialTheme.typography.titleLarge
                )

                Text(
                    text = modelStatus,
                    style =
                        MaterialTheme.typography.bodyMedium
                )

                if (importing) {

                    LinearProgressIndicator(
                        progress = { importFraction },
                        modifier =
                            Modifier.fillMaxWidth()
                    )

                    Text(
                        text = importText,
                        style =
                            MaterialTheme.typography.bodySmall
                    )

                    OutlinedButton(
                        onClick = {
                            cancelImport.set(true)
                            importText = "Annulation..."
                        },
                        modifier =
                            Modifier.fillMaxWidth()
                    ) {
                        Text("Annuler l'import")
                    }

                } else {

                    Button(
                        onClick = {
                            folderLauncher.launch(null)
                        },
                        modifier =
                            Modifier.fillMaxWidth()
                    ) {
                        Text("Importer les modèles (dossier)")
                    }

                    OutlinedButton(
                        onClick = {
                            filesLauncher.launch(arrayOf("*/*"))
                        },
                        modifier =
                            Modifier.fillMaxWidth()
                    ) {
                        Text("Importer des fichiers...")
                    }

                    Text(
                        text =
                            "Choisissez le dossier qui contient les fichiers " +
                            "exportés (nllb_*.onnx, matoub_*.onnx, " +
                            "fadhma_300m_prepared.onnx...). Depuis Android 11, " +
                            "le dossier Download lui-même est refusé : mettez " +
                            "les fichiers dans un sous-dossier, ou utilisez " +
                            "« Importer des fichiers ». Si la traduction a déjà " +
                            "servi, relancez l'application après l'import.",
                        style =
                            MaterialTheme.typography.bodySmall
                    )
                }

                Text(
                    text = "État : $status",
                    style =
                        MaterialTheme.typography.bodyMedium
                )

                Text(
                    text =
                        "Audio : PCM 16 bits, mono, 16 kHz",

                    style =
                        MaterialTheme.typography.bodySmall
                )

                Text(
                    text =
                        "Moteurs prévus : Matoub-82M (TTS) et Fadhma-300M (STT).",

                    style =
                        MaterialTheme.typography.bodySmall
                )
            }
        }
    }




}
