"""Export de NLLB-200 (M2M100) en ONNX pour la traduction français -> kabyle
sur Android.

Fichiers écrits dans NLLB_OUT_DIR :

  nllb_encoder.onnx    inputs_embeds [1, T, d]  attention_mask [1, T]
                       -> last_hidden_state [1, T, d]
  nllb_decoder.onnx    decoder_embeds [1, S, d]  encoder_hidden_states
                       [1, T, d]  encoder_attention_mask [1, T]
                       -> hidden [1, d] pour la dernière position
  nllb_head.onnx       hidden [1, d] -> logits [1, V]  (tête de sortie,
                       à part pour rester sous 2 Go par fichier)
  nllb_embed.i8        matrice d'embeddings V x d en int8 (déjà multipliée
                       par le facteur d'échelle du modèle)
  nllb_embed.scales    V flottants 32 bits little-endian (un par ligne)
  nllb_vocab.tsv       ligne i = "score<TAB>pièce" du jeton i
  nllb_config.txt      paramètres (clé=valeur)

Les embeddings sont hors des graphes ONNX : l'application les lit dans
nllb_embed.i8. Cela évite de dupliquer 1 Go de poids dans l'encodeur et le
décodeur. Les graphes n'ont ni cache clé/valeur : le décodeur est rejoué à
chaque pas, avec les logits calculés seulement pour la dernière position.

Le script compare ensuite, phrase par phrase, la traduction gloutonne
d'ONNX Runtime avec celle de PyTorch (generate, num_beams=1).
"""

import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

MODEL = os.environ.get("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
OUT_DIR = os.path.expanduser(os.environ.get("NLLB_OUT_DIR", "~/kabyle-models/nllb"))
QUANTIZE = os.environ.get("NLLB_QUANT", "1") == "1"
LIMIT = 120  # jetons générés, langue forcée comprise (comme generate)
SRC_LANG = os.environ.get("NLLB_SRC", "fra_Latn")
TGT_LANG = os.environ.get("NLLB_TGT", "kab_Latn")

TESTS = [
    "Bonjour, comment allez-vous ?",
    "Je voudrais un verre d'eau, s'il vous plaît.",
    "Il faut protéger notre langue et notre culture.",
    "Où est la gare ?",
    "Ouvre la porte, il fait très froid dehors.",
    "Merci beaucoup pour votre aide, que Dieu vous protège.",
]


# Phrases dont la tokenisation sert de référence pour tools/nllb/TokenizerCheck.
TOKEN_TESTS = [
    "Bonjour, comment allez-vous ?",
    "Je voudrais un verre d'eau, s'il vous plaît.",
    "Où est la gare ?",
    "Ma mère prépare le couscous pour la fête.",
    "Demain, nous irons au village de mes grands-parents.",
    "Le professeur explique la leçon aux élèves.",
    "Mon frère travaille à Alger depuis trois ans.",
    "Pourquoi n'es-tu pas venu hier soir ?",
    "Nous avons planté des oliviers derrière la maison.",
    "Elle a acheté une robe kabyle pour le mariage de sa sœur.",
    "Tanemmirt aṭas ɣef tallalt-nwen, ad kwen-yeḥrez Rebbi.",
    "Ilaq ad neḥrez tutlayt-nneɣ d yidles-nneɣ.",
    "Iwacu ur d-tusiḍ ara iḍelli ?",
    "Arraw tturaren deg uɣerbaz n taddart.",
    "L\u2019été, on mange des figues \u00ab très mûres \u00bb à 15 h 30 \u2013 ça coûte 3,50 \u20ac.",
    "Ligne un\nligne deux",
    "  Espaces   multiples \t et tabulation.  ",
    "Le café est prêt\u00a0: viens vite !",
    "\ufb01n du \u00bd monde \u2460\u2461",
    "Ça alors\u2026 vraiment ?! Oui\u2026 non.",
    "100% sûr, n\u00b05, M. Amrouche (1913-1976).",
]


def step(title):
    print(f"\n=== {title} ===", flush=True)


step("CHARGEMENT")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang=SRC_LANG)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()
cfg = model.config
d_model = cfg.d_model
vocab_size = model.get_input_embeddings().num_embeddings
embed_scale = float(d_model ** 0.5) if getattr(cfg, "scale_embedding", True) else 1.0

src_id = tokenizer.convert_tokens_to_ids(SRC_LANG)
tgt_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
eos_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id
unk_id = tokenizer.unk_token_id
start_id = cfg.decoder_start_token_id
print(
    f"d_model={d_model} vocab={vocab_size} échelle={embed_scale:.3f} "
    f"{SRC_LANG}={src_id} {TGT_LANG}={tgt_id} eos={eos_id} pad={pad_id} "
    f"départ décodeur={start_id}",
    flush=True,
)
for name, value in (("src", src_id), ("tgt", tgt_id)):
    if value is None or value == unk_id:
        sys.exit(f"Code de langue {name} absent du tokenizer.")

os.makedirs(OUT_DIR, exist_ok=True)


step("VOCABULAIRE ET EMBEDDINGS")

import json  # noqa: E402



def tokenizer_json():
    """Le tokenizer.json, quelle que soit la version de transformers."""
    for holder in ("backend_tokenizer", "_tokenizer"):
        inner = getattr(tokenizer, holder, None)
        if inner is not None and hasattr(inner, "to_str"):
            return json.loads(inner.to_str())
    path = os.path.join(MODEL, "tokenizer.json")
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(MODEL, "tokenizer.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


backend = tokenizer_json()
unigram = backend["model"]
if unigram.get("type") != "Unigram":
    sys.exit(f"Tokenizer {unigram.get('type')} : seul Unigram est géré.")
pieces = unigram["vocab"]

# Table id -> pièce complète (les codes de langue sont des jetons ajoutés).
table = [None] * vocab_size
for index, (piece, score) in enumerate(pieces):
    if index < vocab_size:
        table[index] = (piece, float(score))
specials = set()
for added in backend.get("added_tokens", []):
    if added["id"] < vocab_size:
        table[added["id"]] = (added["content"], 0.0)
        specials.add(added["id"])
for index, entry in enumerate(table):
    if entry is None:
        table[index] = (f"<extra_{index}>", 0.0)
        specials.add(index)
specials.update({tokenizer.bos_token_id, eos_id, pad_id, unk_id})

with open(os.path.join(OUT_DIR, "nllb_vocab.tsv"), "w", encoding="utf-8", newline="\n") as f:
    for piece, score in table:
        if "\t" in piece or "\n" in piece:
            piece = "<?>"
        f.write(f"{score!r}\t{piece}\n")

weights = model.get_input_embeddings().weight.detach().float().numpy()
scaled = weights * embed_scale
row_max = np.abs(scaled).max(axis=1)
scales = np.where(row_max > 0, row_max / 127.0, 1.0).astype("<f4")
quant = np.clip(np.round(scaled / scales[:, None]), -127, 127).astype(np.int8)
quant.tofile(os.path.join(OUT_DIR, "nllb_embed.i8"))
scales.tofile(os.path.join(OUT_DIR, "nllb_embed.scales"))
err = np.abs(quant.astype(np.float32) * scales[:, None] - scaled).max()
print(f"embeddings int8 : écart max {err:.4f} (amplitude {np.abs(scaled).max():.2f})", flush=True)

with open(os.path.join(OUT_DIR, "nllb_config.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write(f"d_model={d_model}\n")
    f.write(f"vocab_size={vocab_size}\n")
    f.write(f"unk_id={unk_id}\n")
    f.write(f"eos_id={eos_id}\n")
    f.write(f"pad_id={pad_id}\n")
    f.write(f"decoder_start_id={start_id}\n")
    f.write(f"src_lang_id={src_id}\n")
    f.write(f"tgt_lang_id={tgt_id}\n")
    f.write("specials=" + ",".join(str(i) for i in sorted(specials)) + "\n")


with open(os.path.join(OUT_DIR, "nllb_tokens_ref.tsv"), "w", encoding="utf-8", newline="\n") as f:
    for text in TESTS + TOKEN_TESTS:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        line = text.replace("\n", "\\n").replace("\t", "\\t")
        f.write(line + "\t" + " ".join(map(str, ids)) + "\n")


class EncoderGraph(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.encoder = m.get_encoder()

    def forward(self, inputs_embeds, attention_mask):
        return self.encoder(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask
        ).last_hidden_state


class DecoderGraph(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.decoder = m.get_decoder()

    def forward(self, decoder_embeds, encoder_hidden_states, encoder_attention_mask):
        hidden = self.decoder(
            inputs_embeds=decoder_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
        ).last_hidden_state
        return hidden[:, -1, :]


class HeadGraph(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.lm_head = m.lm_head

    def forward(self, hidden):
        return self.lm_head(hidden)


def embed(ids):
    """Comme l'application : lignes int8 * échelle par ligne."""
    ids = np.asarray(ids)
    return (quant[ids].astype(np.float32) * scales[ids][:, None])[None]


encoder = EncoderGraph(model).eval()
decoder = DecoderGraph(model).eval()
head = HeadGraph(model).eval()


def encode_source(text):
    body = tokenizer(text, add_special_tokens=False).input_ids
    return [src_id] + body + [eos_id]


step("EXPORT ONNX")

enc_path = os.path.join(OUT_DIR, "nllb_encoder.onnx")
dec_path = os.path.join(OUT_DIR, "nllb_decoder.onnx")
head_path = os.path.join(OUT_DIR, "nllb_head.onnx")

ids = encode_source(TESTS[0])
ex_embeds = torch.from_numpy(embed(ids))
ex_mask = torch.ones(1, len(ids), dtype=torch.long)
with torch.no_grad():
    ex_hidden = encoder(ex_embeds, ex_mask)
ex_dec = torch.from_numpy(embed([start_id, tgt_id, tgt_id + 0]))

src_dim = torch.export.Dim("src", min=2, max=400)
dst_dim = torch.export.Dim("dst", min=2, max=400)

try:
    with torch.no_grad():
        print("Encodeur...", flush=True)
        torch.onnx.export(
            encoder, (ex_embeds, ex_mask),
            input_names=["inputs_embeds", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_shapes={
                "inputs_embeds": {1: src_dim},
                "attention_mask": {1: src_dim},
            },
            dynamo=True, opset_version=18, optimize=False,
        ).save(enc_path)

        print("Décodeur...", flush=True)
        torch.onnx.export(
            decoder, (ex_dec, ex_hidden, ex_mask),
            input_names=["decoder_embeds", "encoder_hidden_states", "encoder_attention_mask"],
            output_names=["hidden"],
            dynamic_shapes={
                "decoder_embeds": {1: dst_dim},
                "encoder_hidden_states": {1: src_dim},
                "encoder_attention_mask": {1: src_dim},
            },
            dynamo=True, opset_version=18, optimize=False,
        ).save(dec_path)

        print("Tête de sortie...", flush=True)
        torch.onnx.export(
            head, (torch.zeros(1, d_model),),
            input_names=["hidden"], output_names=["logits"],
            dynamo=True, opset_version=18, optimize=False,
        ).save(head_path)
except Exception:
    import traceback
    traceback.print_exc()
    sys.exit("ECHEC de l'export ONNX")


def refresh_shapes(path):
    import onnx
    graph = onnx.load(path)
    del graph.graph.value_info[:]
    for value in graph.graph.output:
        value.type.tensor_type.ClearField("shape")
    dynamic = {
        "inputs_embeds": {1: "src"}, "attention_mask": {1: "src"},
        "decoder_embeds": {1: "dst"}, "encoder_hidden_states": {1: "src"},
        "encoder_attention_mask": {1: "src"},
    }
    for value in graph.graph.input:
        for index, name in dynamic.get(value.name, {}).items():
            dim = value.type.tensor_type.shape.dim[index]
            dim.Clear()
            dim.dim_param = name
    graph = onnx.shape_inference.infer_shapes(graph)
    onnx.save(graph, path)


for path in (enc_path, dec_path, head_path):
    refresh_shapes(path)

if QUANTIZE:
    step("QUANTIFICATION INT8")
    from onnxruntime.quantization import QuantType, quantize_dynamic
    for path in (enc_path, dec_path, head_path):
        tmp = path + ".int8"
        quantize_dynamic(path, tmp, weight_type=QuantType.QInt8, per_channel=True)
        os.replace(tmp, path)
        for extra in (tmp + ".data",):
            if os.path.exists(extra):
                os.remove(extra)
for path in (enc_path, dec_path, head_path):
    print(f"{os.path.basename(path)} : {os.path.getsize(path) / 1e6:.0f} Mo", flush=True)


step("VERIFICATIONS")

# 0. Les identifiants source construits ici sont ceux du tokenizer Hugging Face.
mismatch = [t for t in TESTS if tokenizer(t).input_ids != encode_source(t)]
if mismatch:
    print(
        f"0. ATTENTION : {len(mismatch)}/{len(TESTS)} phrases ont des identifiants "
        f"source différents de tokenizer(texte) ; exemple : {mismatch[0]!r}",
        flush=True,
    )
    print("   attendu :", tokenizer(mismatch[0]).input_ids[:12], flush=True)
    print("   obtenu  :", encode_source(mismatch[0])[:12], flush=True)
else:
    print("0. identifiants source identiques à tokenizer(texte)", flush=True)

import onnxruntime as ort  # noqa: E402

enc_session = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
dec_session = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
head_session = ort.InferenceSession(head_path, providers=["CPUExecutionProvider"])


def fp_embed(ids):
    return torch.from_numpy(scaled[np.asarray(ids)].astype(np.float32))[None]


def onnx_logits(hidden, mask, out):
    last = dec_session.run(
        None,
        {
            "decoder_embeds": embed(out),
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": mask,
        },
    )[0]
    return head_session.run(None, {"hidden": last})[0][0]


# 1. Mes enveloppes reproduisent le modèle Hugging Face (embeddings non quantifiés).
worst = 0.0
for text in TESTS:
    ids = encode_source(text)
    out = [start_id, tgt_id, tgt_id]
    with torch.no_grad():
        ref = model(
            input_ids=torch.tensor([ids]), decoder_input_ids=torch.tensor([out])
        ).logits[0, -1]
        mask = torch.ones(1, len(ids), dtype=torch.long)
        hidden = encoder(fp_embed(ids), mask)
        mine = head(decoder(fp_embed(out), hidden, mask))[0]
    worst = max(worst, float((ref - mine).abs().max()))
print(f"1. enveloppes / modèle HF : écart max des logits {worst:.2e}", flush=True)
if worst > 1e-3:
    sys.exit("ECHEC : les enveloppes ne reproduisent pas le modèle.")

# 2. ONNX reproduit les enveloppes PyTorch (mêmes embeddings quantifiés).
worst = 0.0
agree = 0
generated = {}
for text in TESTS:
    ids = encode_source(text)
    mask_np = np.ones((1, len(ids)), dtype=np.int64)
    mask = torch.from_numpy(mask_np)
    hidden_o = enc_session.run(
        None, {"inputs_embeds": embed(ids), "attention_mask": mask_np}
    )[0]
    with torch.no_grad():
        hidden_t = encoder(torch.from_numpy(embed(ids)), mask)
    worst = max(worst, float(np.abs(hidden_o - hidden_t.numpy()).max()))

    out_o = [start_id, tgt_id]
    out_t = [start_id, tgt_id]
    for _ in range(LIMIT - 1):
        lo = onnx_logits(hidden_o, mask_np, out_o)
        with torch.no_grad():
            lt = head(decoder(torch.from_numpy(embed(out_t)), hidden_t, mask))[0].numpy()
        worst = max(worst, float(np.abs(lo - lt).max()))
        out_o.append(int(np.argmax(lo)))
        out_t.append(int(np.argmax(lt)))
        if out_o[-1] == eos_id and out_t[-1] == eos_id:
            break
        if out_o[-1] == eos_id or out_t[-1] == eos_id:
            break
    agree += out_o == out_t
    generated[text] = out_o
print(
    f"2. ONNX / enveloppes : écart max {worst:.2e}, "
    f"{agree}/{len(TESTS)} traductions gloutonnes identiques",
    flush=True,
)

# 3. La boucle gloutonne (départ du décodeur, langue forcée, arrêt) est celle
#    de generate() : même chose en 32 bits, sans quantification.
strict = 0
info = 0
for text in TESTS:
    ids = torch.tensor([encode_source(text)])
    with torch.no_grad():
        ref = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            forced_bos_token_id=tgt_id, num_beams=1, do_sample=False,
            max_new_tokens=LIMIT,
        )[0].tolist()
        mask = torch.ones(1, ids.shape[1], dtype=torch.long)
        hidden = encoder(fp_embed(ids[0].tolist()), mask)
        out = [start_id, tgt_id]
        for _ in range(LIMIT - 1):
            nxt = int(head(decoder(fp_embed(out), hidden, mask))[0].argmax())
            out.append(nxt)
            if nxt == eos_id:
                break
    strict += out == ref
    info += generated[text] == ref
    body = [i for i in generated[text] if i not in specials]
    print("  ", tokenizer.decode(body)[:120], flush=True)
print(f"3. boucle gloutonne 32 bits / generate() : {strict}/{len(TESTS)} identiques", flush=True)
print(f"   ONNX quantifié / generate() : {info}/{len(TESTS)} identiques (indicatif)", flush=True)
if strict < len(TESTS):
    sys.exit("ECHEC : la boucle gloutonne diffère de generate().")

ok = worst < (0.5 if QUANTIZE else 1e-2) and agree >= len(TESTS) - (2 if QUANTIZE else 0)
if not ok:
    sys.exit("ECHEC : l'ONNX s'écarte trop de PyTorch.")
print("\n=== EXPORT TERMINE ===", flush=True)
print(f"Dossier : {OUT_DIR}", flush=True)
