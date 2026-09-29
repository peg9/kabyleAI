"""Écrit la tokenisation de référence de phrases de test.

Usage : python tokens_ref.py NLLB_MODEL sortie.tsv [phrases.txt]
NLLB_MODEL : identifiant Hugging Face ou dossier local. Sans fichier de
phrases, une série de phrases françaises et kabyles est utilisée.
Ensuite : java TokenizerCheck nllb_config.txt nllb_vocab.tsv sortie.tsv
"""

import sys
import warnings

warnings.filterwarnings("ignore")
from transformers import AutoTokenizer

DEFAULT = [
    "Bonjour, comment allez-vous ?",
    "Je voudrais un verre d'eau, s'il vous plaît.",
    "Où est la gare ?",
    "Il pleut depuis ce matin.",
    "Ma mère prépare le couscous pour la fête.",
    "Demain, nous irons au village de mes grands-parents.",
    "Le professeur explique la leçon aux élèves.",
    "J'aime beaucoup la montagne en hiver.",
    "Combien coûte ce pain ?",
    "Mon frère travaille à Alger depuis trois ans.",
    "Il faut protéger notre langue et notre culture.",
    "Pourquoi n'es-tu pas venu hier soir ?",
    "Ouvre la porte, il fait très froid dehors.",
    "Nous avons planté des oliviers derrière la maison.",
    "Merci beaucoup pour votre aide, que Dieu vous protège.",
    "Tanemmirt aṭas ɣef tallalt-nwen, ad kwen-yeḥrez Rebbi.",
    "Ilaq ad neḥrez tutlayt-nneɣ d yidles-nneɣ.",
    "Iwacu ur d-tusiḍ ara iḍelli ?",
    "Arraw tturaren deg uɣerbaz n taddart.",
    "L’été, on mange des figues « très mûres » à 15 h 30 – ça coûte 3,50 €.",
    "Ligne un\nligne deux",
    "  Espaces   multiples \t et tabulation.  ",
    "Le café est prêt\u00a0: viens vite !",
    "ﬁn du ½ monde ①②",
    "Ça alors… vraiment ?! Oui… non.",
    "100% sûr, n°5, M. Amrouche (1913-1976).",
]

model, out = sys.argv[1], sys.argv[2]
texts = DEFAULT
if len(sys.argv) > 3:
    texts = [l.rstrip("\n") for l in open(sys.argv[3], encoding="utf-8") if l.strip()]
tok = AutoTokenizer.from_pretrained(model)
with open(out, "w", encoding="utf-8") as f:
    for text in texts:
        ids = tok(text, add_special_tokens=False).input_ids
        f.write(text.replace("\n", "\\n").replace("\t", "\\t") + "\t" + " ".join(map(str, ids)) + "\n")
print(len(texts), "phrases écrites dans", out)
