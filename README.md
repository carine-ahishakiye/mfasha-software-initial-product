# MFASHA – Kinyarwanda Agricultural Chatbot

**MFASHA** ("helper") is a Kinyarwanda-language chatbot that answers farmers' questions on crops, livestock, pests and diseases, and farm inputs/services. It combines a **topic classifier** with a **hybrid retrieval engine** over a knowledge base of 971 agricultural passages, and is served through a **Gradio web interface**.

> **Track:** Machine Learning – Initial Software Product / MVP demonstration


## 1. Description

Many Rwandan farmers need quick, trustworthy agricultural advice in Kinyarwanda, but extension officers are limited and most digital resources are in English. MFASHA lets a user type a question in Kinyarwanda (for example *"Inka ifite impiswi navura nte?"*) and returns the most relevant answer from a curated knowledge base.

**How it works (pipeline):**
1. **Clean** the question (removes greetings/self-introductions such as *"Muraho, nitwa … ndashaka kumenya …"*).
2. **Classify** the topic (animal health, crop pests and diseases, crop production, inputs and services, livestock management).
3. **Retrieve** the best passages by fusing TF-IDF (body + title), stemmed BM25, a topic boost and an animal-species mismatch penalty.
4. **Respond** with the answer, a list of close matches, or a "no answer found" message if confidence is too low. Unanswered questions are logged to `unanswered.csv` so the knowledge base can be improved.

## 2. GitHub Repository

🔗 **Repo link:** `https://github.com/carine-ahishakiye/mfasha-software-initial-product.git`
## Video Demo

🎥 **Link:** `https://youtu.be/Dk9rJaVx2ZI`
## 3. Dataset

- **Source:** [`C4IR-RW/kinya-ag-retrieval`](https://huggingface.co/datasets/C4IR-RW/kinya-ag-retrieval) on Hugging Face (Kinyarwanda agricultural retrieval dataset).
- **Content:** (question, positive passage, negative passage) triplets, question texts and answer passages.
- **Size used:** 19,537 questions linked to 971 distinct answer passages.
- Dataset files used: `rw_ag_retrieval_qpntriplets_{train,dev,test}.tsv`, `rw_ag_retrieval_questions.tsv`, `rw_ag_retrieval_answers.tsv`.
- The dataset has **no topic labels**, so labels were created with keyword rules on passage titles and then corrected manually (see below).

## 4. Data Visualization & Data Engineering

**Data engineering steps** (all in `MFASHA_Chatbot.ipynb`):

| Step | What was done |
|---|---|
| Loading & validation | Downloaded files from Hugging Face; asserted expected column counts and the 19,537 question total; verified merges with `validate=` |
| ID parsing | Parsed IDs like `Q00487T001` / `A020T001` to link each question to its passage (`q_T`) |
| Data quality check | Passage length, missing titles, very short (heading-only) passages, duplicate text |
| Topic labelling | Keyword rules on passage titles + manual exception rules + 20 manual overrides, producing 5 categories |
| Question cleaning | Removed greeting + self-introduction (affects ~18% of questions); never returns fewer than 3 words |
| Leak-free splitting | `StratifiedGroupKFold` grouped by passage id so no passage appears in more than one of train/dev/test; asserts overlap = 0 |

**Visualizations produced** (saved as PNGs by the notebook):

| Figure | Shows |
|---|---|
| `fig_data_overview.png` | (A) questions per category by split, (B) passage length distribution, (C) effect of greeting cleaning |
| `fig_question_length.png` | Question length (words) before vs after cleaning |
| `fig_confusion_matrix.png` | Row-normalised confusion matrix of the topic classifier on the test set |

![Data overview](docs/images/fig_data_overview.png)
![Question length](docs/images/fig_question_length.png)
![Confusion matrix](docs/images/fig_confusion_matrix.png)

## 5. Model Architecture

MFASHA uses classical, lightweight NLP models (no neural network layers or activation functions), which suit low-resource Kinyarwanda and run on GPU. It has two components.

### 5.1 Topic Classifier

```
Cleaned question (lower-cased)
        │
        ▼
FeatureUnion
 ├─ TF-IDF on words        (1–2 grams, sublinear TF, min_df=2)
 └─ TF-IDF on characters   (char_wb 2–5 grams, sublinear TF, min_df=2)
        │
        ▼
LinearSVC (C=0.5, class_weight="balanced")
        │
        ▼
Topic ∈ {animal health, crop pests and diseases, crop production (general),
         inputs and services, livestock management}
```

- **Optimization:** LinearSVC (squared hinge loss, L2 regularisation, `C=0.5`) with balanced class weights for the imbalanced classes.
- **Why character n-grams:** they handle Kinyarwanda's rich word endings and spelling variants.
- **Model selection:** compared a most-frequent baseline, Logistic Regression (C=5), LinearSVC, and unweighted LinearSVC on the **dev set only** by macro-F1. Test was scored once.

### 5.2 Hybrid Retrieval Engine

```
Question ──► Topic classifier ──► predicted topic
   │
   ├─ TF-IDF similarity vs passage BODY      (word 1–2g + char 3–5g)   weight 1.0
   ├─ TF-IDF similarity vs passage TITLE     (word 1–2g + char 3–5g)   weight 2.0
   ├─ BM25 (Okapi) on 6-char stemmed tokens                            weight 1.0
   ├─ + topic boost (2.0) for passages in the predicted category
   └─ − animal-species mismatch penalty (50) (e.g. question about chickens ≠ cattle passage)
              │
              ▼
        Fused score (each component z-score normalised)
              │
              ▼
   Top-1 body similarity ≥ 0.30 → show answer
   0.15 – 0.30                  → show "possible matches"
   < 0.15                       → "No answer found" (logged)
```

- **Fusion weights** (`w_title=2.0, w_bm25=1.0, boost=2.0`) were chosen by **grid search on the dev set** (maximising MRR).
- **Rejection threshold (0.30)** was chosen by judgement to answer most real questions while rejecting off-topic ones (not the dev-optimal value, which was too strict for users).
- Final index text = passage text + all training questions linked to that passage.

## 6. Initial Performance Metrics

### 6.1 Topic classification (TF-IDF + LinearSVC)

| Split | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Dev | 0.874 | 0.832 | 0.892 | 0.855 |
| **Test** | **0.901** | **0.890** | **0.897** | **0.893** |
| Test – baseline (most frequent) | 0.456 | 0.091 | 0.200 | 0.125 |

Per-class test results (excerpt): animal health F1 0.884 · crop pests and diseases F1 0.881 · crop production (general) F1 0.903 · inputs and services F1 0.844 · livestock management F1 0.953.

**Dev model comparison**

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline (most frequent) | 0.502 | 0.134 |
| TF-IDF + LogReg | 0.869 | 0.848 |
| **TF-IDF + LinearSVC** | **0.874** | **0.855** |
| TF-IDF + LinearSVC (no weights) | 0.870 | 0.850 |

### 6.2 Retrieval (test set: 2,802 questions, 971 passages)

| Method | Hit@1 | Hit@5 | MRR | Category@1 |
|---|---|---|---|---|
| Old: TF-IDF body only | 0.532 | 0.805 | 0.654 | 0.877 |
| Stemmed BM25 only | 0.480 | 0.749 | 0.602 | 0.856 |
| **Chosen fusion** | **0.605** | **0.862** | **0.719** | **0.922** |

### 6.3 Out-of-scope handling (threshold = 0.30)

- Real test questions answered: **83.1%**
- Off-topic questions rejected: **9 of 10**
- Hidden-passage simulation (test): 85.3% of answerable questions accepted, 33.4% of unanswerable rejected, 68% top-1 correct among accepted.

All results are also saved as CSVs: `results_dev.csv`, `results_test.csv`, `results_retrieval.csv`, `results_threshold_tradeoff.csv`, `summary_classification.csv`, `summary_retrieval.csv`.

## 7. Environment & Project Setup

**Requirements:** Python 3.10+ (tested in Google Colab), internet access for the first run (dataset is downloaded from Hugging Face).

```bash
# 1. Clone the repository
git clone https://https://github.com/carine-ahishakiye/mfasha-software-initial-product.git

```

**`requirements.txt`**
```
numpy
pandas
matplotlib
seaborn
scikit-learn
rank_bm25
joblib
huggingface_hub
gradio
jupyter
```

**Run the notebook**
```bash
jupyter notebook MFASHA_Chatbot.ipynb
```
Or open it in Google Colab and choose **Runtime → Run all**. The notebook downloads the dataset, builds the labels, trains and evaluates the models, saves artefacts (`category_model.joblib`, `passages_kb.csv`, `threshold.json`, result CSVs and figures) and launches the Gradio app in the final cell.

## 8. Running the App (MVP)

The MVP is a **web interface built with Gradio**, launched from the last cell of the notebook:

```python
demo.launch(share=True)  
```

**Features demonstrated:**
- Kinyarwanda question box with example questions
- Topic detection and best-answer display (*Igisubizo*)
- "Possible matches" list when confidence is medium (*Ibishobora kuba bihuye*)
- "No answer found" message for out-of-scope questions (*Nta gisubizo cyabonetse*), with the question logged to `unanswered.csv`
- Safety note advising users to consult a veterinarian or agronomist for urgent cases

**Example questions to try**
- `Kurera imishwi bikorwa bite?`
- `Ni gute nahinga inyanya neza?`
- `Inka ifite impiswi navura nte?`
- `Inkoko zirwaye New Castle zivurwa gute?`
- `Gusarura ibitoki bikorwa bite?`


**Navigation & layout:** single-page interface — title and tagline at the top, one question box, one primary button (*Shaka Igisubizo* / "Find answer"), example questions beneath, and the answer area below. Pressing Enter also submits.

## 10. Deployment Plan

| Phase | Plan |
|---|---|
| **1. MVP (current)** | Gradio app run from Colab/local machine with a temporary public share link for demos and supervisor feedback |
| **2. Hosted web app** | Move the code into `app.py`, load saved artefacts (`category_model.joblib`, `passages_kb.csv`, `threshold.json`) and deploy on **Hugging Face Spaces** (free, Gradio-native) or Render |
| **3. API** | Wrap the `ask` logic in a **FastAPI** service (`POST /ask`) with auto-generated **Swagger UI** docs, so mobile apps and other services can use it |
| **4. Mobile / low-bandwidth access** | Mobile-friendly web UI first; later an Android app and a USSD/SMS or WhatsApp gateway for farmers without smartphones |
| **5. Continuous improvement** | Review `unanswered.csv` regularly, add new passages with agricultural experts, have a Kinyarwanda speaker validate the topic labels, and retrain |
| **6. Longer-term** | Evaluate Kinyarwanda neural embedding models, add voice input/output, and partner with MINAGRI/extension services for content validation |



## Repository Structure

```
├── MFASHA_Chatbot.ipynb          # Full pipeline: data, models, evaluation, Gradio app
├── README.md
```



## Acknowledgements

Dataset: C4IR Rwanda – `kinya-ag-retrieval` (Hugging Face).
