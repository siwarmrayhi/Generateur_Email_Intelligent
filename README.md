

# ✉️ Générateur d'Emails

### Fine-tuning d'un Transformer pour la rédaction d'emails professionnels

Un modèle de génération de texte conditionnelle (GPT-2) fine-tuné sur de vrais emails d'entreprise pour rédiger des emails professionnels à partir d'un court contexte et d'un ton choisi.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C)
![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E)
![GPT-2](https://img.shields.io/badge/Model-GPT--2-green)
![License](https://img.shields.io/badge/Usage-Éducatif-lightgrey)

[🎬 Voir la démo →](#-démo)



---

## 📑 Table des matières

- [Aperçu](#-aperçu)
- [Démo](#-démo)
- [Vue d'ensemble du pipeline](#-vue-densemble-du-pipeline)
- [Dataset](#-dataset)
- [Preprocessing](#-preprocessing)
- [Génération conditionnelle : conception du prompt](#-génération-conditionnelle--conception-du-prompt)
- [Modèle et fine-tuning](#-modèle-et-fine-tuning)
- [Méthodologie d'évaluation](#-méthodologie-dévaluation)
- [Résultats](#-résultats)
- [Exemples de sorties](#-exemples-de-sorties)
- [Démarrage rapide](#-démarrage-rapide)
- [Structure du projet](#️-structure-du-projet)
- [Limites et perspectives](#️-limites-et-perspectives)
- [Éthique et confidentialité des données](#-éthique-et-confidentialité-des-données)
- [Références](#-références)
- [Auteure](#-auteure)

---

## 🎯 Aperçu

Rédiger des emails professionnels est répétitif : proposer un rendez-vous, relancer, remercier un client, confirmer un horaire. Ce projet explore jusqu'où un petit modèle de langage open-source peut aller sur cette tâche lorsqu'il est adapté au domaine par *transfer learning*.

### Ce que couvre ce dépôt

- 🧹 Un pipeline NLP de données reproductible (nettoyage, filtrage, étiquetage, découpage) sur ~500k emails réels
- 🧠 Le fine-tuning de GPT-2 (un Transformer décodeur pré-entraîné) comme modèle de langage causal
- 🎛️ La génération conditionnelle : le modèle est guidé par un contexte et un ton via un prompt structuré
- 📏 Un protocole d'évaluation quantitative et qualitative (perplexité, BLEU, ROUGE, revue manuelle)
- 🔌 Un code d'inférence léger pour charger le modèle entraîné et générer des emails

**Concepts illustrés :** NLP · Transformers · Transfer Learning · Fine-tuning · Tokenisation (BPE) · Modélisation de langage causale · Évaluation de modèles

---

## 🎬 Démo

📹 **Vidéo de démonstration :** [Lien Google Drive](https://drive.google.com/drive/folders/12PJW3JyVrumBnSf5JPkVOJ7uqHP5tBAT?usp=sharing) 

La démo montre le modèle entraîné générant des emails pour différents contextes et tons, et compare la sortie du GPT-2 de base à celle du modèle fine-tuné.

**Exemple d'interaction** (la sortie est en anglais, langue du dataset) :

```text
Contexte : Proposer un rendez-vous à un client la semaine prochaine
Ton      : formel

→ Email généré :
  Dear Mr. Dupont,
  I hope you are well. I would like to propose a meeting next week to ...
```

---

## 🔭 Vue d'ensemble du pipeline

```text
📦 Enron emails.csv (~500k messages bruts)
        │
        ▼
🧹 Nettoyage et filtrage
        │
        ▼
🏷️ Étiquetage heuristique (ton + contexte)
        │
        ▼
🧾 Formatage du prompt (CONTEXT / TONE / EMAIL)
        │
        ▼
🔤 Tokenisation (GPT-2 BPE)
        │
        ▼
🧠 Fine-tuning (LM causal · PyTorch + HF Trainer)
        │
        ▼
📏 Évaluation (PPL · BLEU · ROUGE · revue)
        │
        ▼
💾 Modèle sauvegardé + inférence
```

---

## 📦 Dataset

| Propriété | Valeur |
|---|---|
| **Nom** | Enron Email Dataset |
| **Source** | [Kaggle – Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset) |
| **Contenu** | Emails d'entreprise réels (publication publique) |
| **Taille brute** | ~500k messages |
| **Langue** | Anglais |
| **Après nettoyage** | _à renseigner : N emails_ |
| **Découpage** | 80 % entraînement / 10 % validation / 10 % test (seed fixée) |

> **Pourquoi Enron ?** C'est l'une des plus grandes collections publiques d'emails professionnels réels, ce qui la rend adaptée pour apprendre la structure et le registre de la communication en entreprise.

---

## 🧹 Preprocessing

Les emails bruts contiennent beaucoup de bruit qu'un modèle de langage apprendrait volontiers à imiter. Le pipeline (`data/prepare_data.py`) applique :

| Étape | Ce qu'elle fait | Pourquoi |
|---|---|---|
| **Suppression des fils** | Retire les blocs transférés / réponses (`-----Original Message-----`, `Forwarded by …`) | Évite d'apprendre les en-têtes et citations |
| **Anonymisation** | Supprime URLs, adresses email et numéros de téléphone (regex) | Confidentialité + moins de bruit |
| **Normalisation** | Réduit les espaces et lignes de séparation | Tokens plus propres |
| **Filtrage par longueur** | Garde les emails de 50 à 2 000 caractères | Retire les valeurs aberrantes |
| **Déduplication** | Supprime les doublons exacts | Évite la mémorisation et la fuite train/test |
| **Étiquetage** | Attribue un ton et un contexte par règles de mots-clés | Crée le signal de conditionnement |
| **Découpage** | Split 80/10/10 mélangé avec seed fixée | Évaluation reproductible |

**Approche d'étiquetage.** Le ton (*formel / neutre / informel*) et le contexte (ex. *proposer un rendez-vous, demander des informations, remercier, faire un suivi*) sont déduits par des heuristiques de mots-clés, sans annotation humaine. C'est une stratégie pragmatique de supervision faible ; ses limites sont discutées dans [Limites et perspectives](#️-limites-et-perspectives).

---

## 🧾 Génération conditionnelle : conception du prompt

Un modèle de langage classique ne fait que continuer un texte. Pour le rendre contrôlable, chaque exemple d'entraînement est sérialisé ainsi :

```text
[CONTEXT] {contexte} [TONE] {ton} [EMAIL] {corps de l'email}
```

À l'inférence, seul le préfixe est fourni et le modèle complète l'email :

```text
[CONTEXT] Proposer un rendez-vous [TONE] formel [EMAIL]  →  "Dear ..., I would like to ..."
```

C'est une forme simple et efficace de **conditionnement par prompt** : le modèle apprend l'association entre les balises de contrôle et le style/contenu qui les suit.

---

## 🧠 Modèle et fine-tuning

### Pourquoi GPT-2 + transfer learning ?

| Option | Verdict |
|---|---|
| Entraîner un Transformer *from scratch* | ❌ Demande bien plus de données et de calcul que disponible |
| Utiliser un modèle via API propriétaire | ❌ Aucun contrôle sur l'entraînement ; ce n'est pas l'objectif du projet |
| Fine-tuner un modèle open-source pré-entraîné (GPT-2) | ✅ Rapide, reproductible, tourne sur un GPU Colab gratuit |

GPT-2 (≈124M paramètres) est un *modèle de langage*, pas un grand modèle de langage au sens actuel du terme — ce qui en fait justement un bon terrain d'expérimentation : assez petit pour être fine-tuné de bout en bout sur un seul GPU T4, tout en montrant clairement l'adaptation au domaine.

### Configuration d'entraînement

| Hyperparamètre | Valeur |
|---|---|
| Modèle de base | `gpt2` (124M) |
| Objectif | Modélisation de langage causale (prédiction du token suivant) |
| Longueur max de séquence | 512 tokens |
| Époques | 3 |
| Taille de batch | 4 (8 sur Colab T4 — _adapte à ce que tu as utilisé_) |
| Learning rate | 5e-5 |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Précision mixte | fp16 si un GPU est disponible |
| Early stopping | patience = 3 évaluations |
| Évaluation / sauvegarde | toutes les 500 / 1 000 étapes, meilleur checkpoint conservé |
| Frameworks | PyTorch · 🤗 Transformers `Trainer` · 🤗 Datasets |
| Matériel | _à renseigner : GPU T4 Google Colab_ |
| Durée d'entraînement | _à renseigner_ |

### Déroulement de l'entraînement

1. **Tokenisation** avec le tokenizer byte-pair-encoding de GPT-2 (token de padding = EOS).
2. **Collation** des données avec `DataCollatorForLanguageModeling(mlm=False)` — les labels sont les entrées décalées d'un token.
3. **Optimisation** de la cross-entropy avec AdamW, warmup linéaire et weight decay.
4. **Sélection du modèle** : le checkpoint avec la plus faible loss de validation est conservé.

### Stratégie de décodage

La génération utilise l'**échantillonnage** (`temperature=0.7`, `top_p=0.9`, `top_k=50`) plutôt que le décodage glouton, ce qui donne des emails plus naturels et moins répétitifs.

---

## 📏 Méthodologie d'évaluation

Une seule métrique ne suffit jamais pour la génération de texte ; l'évaluation combine donc plusieurs angles :

| Niveau | Métrique | Ce qu'elle apporte |
|---|---|---|
| **Modélisation de langage** | Loss et perplexité (`exp(loss)`) de validation / test | Capacité à prédire des emails jamais vus |
| **Recouvrement avec les références** | BLEU, ROUGE-1/2/L sur des échantillons générés vs. vrais emails | Similarité de surface (utile pour comparer, limitée pour du texte ouvert) |
| **Comparaison à une baseline** | GPT-2 de base vs. GPT-2 fine-tuné sur le même jeu de test | Prouve que le fine-tuning a réellement aidé |
| **Revue qualitative** | Vérification manuelle de la cohérence, de la structure (salutation → corps → formule de politesse) et du respect du ton | Ce que les métriques automatiques ne voient pas |

> ⚠️ BLEU/ROUGE comparent à **un seul** email de référence, alors que plusieurs emails différents peuvent être tout aussi bons pour un même contexte. Ils sont rapportés pour la comparabilité, pas comme mesure de « vérité ».

---

## 📊 Résultats

> Remplace les champs ci-dessous par les valeurs produites par `models/evaluate.py` sur ton jeu de test.

| Métrique | GPT-2 de base | GPT-2 fine-tuné |
|---|---|---|
| Loss de test | _à renseigner_ | _à renseigner_ |
| Perplexité ↓ | _à renseigner_ | _à renseigner_ |
| BLEU ↑ | _à renseigner_ | _à renseigner_ |
| ROUGE-L ↑ | _à renseigner_ | _à renseigner_ |

**Courbes d'entraînement**

<!-- ![Courbes de loss](assets/loss_curves.png) -->

### Observations clés

> _Écris 3 constats honnêtes après avoir regardé tes résultats, par exemple :_

- La perplexité passe de **X** à **Y** après fine-tuning → le modèle s'adapte au style des emails.
- Le modèle respecte le ton demandé dans **N** échantillons sur **M** relus manuellement.
- Cas d'échec typiques : _répétitions / formulations génériques / contexte ignoré_.

---

## 🧪 Exemples de sorties

<details>
<summary><b>GPT-2 de base vs. fine-tuné (cliquer pour déplier)</b></summary>

**Prompt :** `[CONTEXT] Proposer un rendez-vous [TONE] formel [EMAIL]`

| GPT-2 de base | GPT-2 fine-tuné |
|---|---|
| _coller la vraie sortie_ | _coller la vraie sortie_ |

**Prompt :** `[CONTEXT] Remercier un client [TONE] informel [EMAIL]`

| GPT-2 de base | GPT-2 fine-tuné |
|---|---|
| _coller la vraie sortie_ | _coller la vraie sortie_ |

</details>

---

## 🚀 Démarrage rapide

### 1. Installation

```bash
git clone https://github.com/<ton-username>/email-ai-generator.git
cd email-ai-generator

python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Préparer les données

```bash
# Télécharge emails.csv depuis Kaggle et place-le dans data/raw/
python data/prepare_data.py
```

### 3. Fine-tuner

```bash
python models/train.py
```

> 💡 **Pas de GPU en local ?** Exécute les mêmes scripts dans Google Colab (*Exécution → Modifier le type d'exécution → GPU*). Le fine-tuning sur un T4 est nettement plus rapide que sur CPU.

### 4. Évaluer

```bash
python models/evaluate.py
```

### 5. Générer un email

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "./models/saved_model"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path).eval()

prompt = "[CONTEXT] Proposer un rendez-vous [TONE] formel [EMAIL]"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True).split("[EMAIL]")[-1].strip())
```

Le modèle entraîné peut aussi être exposé via un petit service **FastAPI** (`backend/app.py`, endpoint `POST /generate`) pour l'intégrer à n'importe quelle application cliente.

---

## 🗂️ Structure du projet

```text
email-ai-generator/
├── data/
│   ├── raw/                  # emails.csv (non versionné)
│   ├── processed/            # train.csv · val.csv · test.csv
│   └── prepare_data.py       # nettoyage, étiquetage, découpage
├── models/
│   ├── train.py              # script de fine-tuning (HF Trainer)
│   ├── evaluate.py           # perplexité, BLEU, ROUGE, exemples
│   └── saved_model/          # poids fine-tunés (non versionnés)
├── backend/
│   └── app.py                # API d'inférence optionnelle (FastAPI)
├── notebooks/
│   └── exploration.ipynb     # EDA et expérimentations
├── assets/                   # miniature de démo, courbes de loss
├── requirements.txt
└── README.md
```

---

## ⚠️ Limites et perspectives

### Limites connues

- 🇬🇧 **Langue :** Enron est uniquement en anglais, le modèle génère donc des emails en anglais. Générer en français ou en arabe demanderait un dataset adapté ou un modèle de base multilingue.
- 🏷️ **Labels faibles :** le ton et le contexte sont déduits par règles de mots-clés, donc les étiquettes sont bruitées et grossières.
- 🛑 **Arrêt de la génération :** sans token de fin de séquence explicite dans les textes d'entraînement, les générations peuvent se poursuivre jusqu'à `max_new_tokens`.
- 🎯 **Loss sur le prompt :** la loss est calculée sur toute la séquence, balises de contrôle comprises, et pas seulement sur le corps de l'email.
- 📉 **Limites des métriques :** BLEU/ROUGE récompensent la similarité avec une seule référence, pas la qualité globale.
- 🧱 **Taille du modèle :** GPT-2 (124M) est petit ; il peut être générique ou répétitif sur les emails longs.

### Perspectives

- [ ] Ajouter un token EOS à chaque exemple d'entraînement
- [ ] Masquer les tokens du prompt dans la loss pour que le modèle n'apprenne que le corps de l'email
- [ ] Déclarer `[CONTEXT]`, `[TONE]`, `[EMAIL]` comme *special tokens*
- [ ] Meilleurs labels : classifieur zero-shot ou annotation assistée par LLM
- [ ] Fine-tuning efficace en paramètres (LoRA) pour un entraînement plus rapide et des checkpoints plus légers
- [ ] Modèles de base plus grands ou multilingues (ex. GPT-2 medium, mT5, Mistral-7B avec LoRA)
- [ ] Évaluation humaine systématique avec une grille (cohérence, pertinence, ton)

---

## 🔐 Éthique et confidentialité des données

- Le corpus Enron est composé d'emails de **personnes réelles**, rendus publics lors de procédures judiciaires. Les URLs, adresses email et numéros de téléphone sont supprimés au preprocessing, mais le dataset peut encore contenir des informations personnelles.
- Les modèles peuvent **mémoriser et reproduire** des fragments de leurs données d'entraînement ; ne déploie pas ce modèle publiquement sans une revue de confidentialité dédiée.
- Les emails générés doivent **toujours être relus par un humain** avant envoi.
- Ce projet est destiné à des **fins pédagogiques et de recherche**.

---

## 📚 Références

1. Vaswani et al., *Attention Is All You Need* (2017)
2. Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019)
3. Papineni et al., *BLEU: a Method for Automatic Evaluation of Machine Translation* (2002)
4. Lin, *ROUGE: A Package for Automatic Evaluation of Summaries* (2004)
5. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)
6. [Documentation Hugging Face Transformers](https://huggingface.co/docs/transformers)
7. [Enron Email Dataset sur Kaggle](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)

---

## 👩‍💻 Auteure

**Siwar Mrayhi**

[LinkedIn](https://linkedin.com/in/siwar-mrayhi ) · [Email](siwar.mrayhi@etudiant-isi.utm.tn) <!-- remplace # par tes liens -->

