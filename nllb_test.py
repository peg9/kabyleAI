"""Test de qualité : traduction français -> kabyle avec NLLB-200 (600M).

Usage : python nllb_test.py [fichier_phrases.txt]
Sans fichier, une vingtaine de phrases de test sont traduites.
Le modèle (environ 2,5 Go) est téléchargé une fois dans HF_HOME.
"""

import os
import sys
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = os.environ.get("NLLB_MODEL", "facebook/nllb-200-distilled-600M")

PHRASES = [
    "Bonjour, comment allez-vous ?",
    "Je voudrais un verre d'eau, s'il vous plaît.",
    "Où est la gare ?",
    "Quelle heure est-il ?",
    "Il pleut depuis ce matin.",
    "Je ne comprends pas ce que tu dis.",
    "Ma mère prépare le couscous pour la fête.",
    "Demain, nous irons au village de mes grands-parents.",
    "Le professeur explique la leçon aux élèves.",
    "J'aime beaucoup la montagne en hiver.",
    "Combien coûte ce pain ?",
    "Mon frère travaille à Alger depuis trois ans.",
    "Les enfants jouent dans la cour de l'école.",
    "Il faut protéger notre langue et notre culture.",
    "Pourquoi n'es-tu pas venu hier soir ?",
    "Ouvre la porte, il fait très froid dehors.",
    "Nous avons planté des oliviers derrière la maison.",
    "Elle a acheté une robe kabyle pour le mariage de sa sœur.",
    "Si tu as faim, il reste du pain et de l'huile d'olive.",
    "Merci beaucoup pour votre aide, que Dieu vous protège.",
]


def load_phrases():
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if lines:
            return lines
    return PHRASES


print(f"Chargement de {MODEL} ...", flush=True)
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang="fra_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()
print(f"Chargé en {time.time() - start:.0f} s", flush=True)

target = tokenizer.convert_tokens_to_ids("kab_Latn")
if target is None or target == tokenizer.unk_token_id:
    sys.exit("Le code de langue kab_Latn est absent du tokenizer.")

for phrase in load_phrases():
    inputs = tokenizer(phrase, return_tensors="pt")
    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=target,
            num_beams=4,
            max_new_tokens=128,
        )
    text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    print(f"\nFR  {phrase}\nKAB {text}\n    ({time.time() - start:.1f} s)", flush=True)
