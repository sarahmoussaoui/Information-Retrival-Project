# Information Retrieval Project: MEDLINE

Comparative evaluation of eleven information retrieval models on the MEDLINE benchmark, with a Learning to Rank (LTR) model and an interactive Streamlit app to explore rankings and metrics.

**Live demo:** https://information-retrival-project-426637717447.us-central1.run.app/
**Full report (French):** [`FinalReport/Rapport final ri.pdf`](FinalReport/Rapport%20final%20ri.pdf)

---

## Overview

The project covers the whole IR pipeline: parsing and preprocessing the collection, building indexes, running several retrieval models, evaluating them with standard metrics, combining them with LTR, and visualizing everything in a web app.

## Dataset

[MEDLINE](https://ir.dcs.gla.ac.uk/resources/test_collections/medl/) test collection (University of Glasgow):

| File | Content |
|------|---------|
| `MED.ALL` | 1,033 medical abstracts (documents) |
| `MED.QRY` | 30 queries |
| `MED.REL` | Relevance judgments (query → relevant documents) |

## Preprocessing and indexing

1. Tokenization (regex), lowercasing
2. Stopword removal
3. Porter stemming
4. Statistics: TF, normalized TF, DF, IDF (`log(N/n_i + 1)`), collection frequencies, document lengths
5. Structures: vocabulary, sparse (CSR) TF-IDF and binary document–term matrices, inverted index

Preprocessed data is cached as JSON to avoid recomputation.

## Retrieval models

| Family | Models |
|--------|--------|
| Vector | VSM (cosine), LSI (k = 100) |
| Probabilistic | BIR and Extended BIR (with / without relevance feedback), BM25 (k1 = 1.2, b = 0.75) |
| Language models | MLE, Laplace (add-1), Jelinek–Mercer (λ = 0.2), Dirichlet (μ = 0.3 × avg. doc length) |

`run_all_models` runs every model on every query and produces the ranked lists used for evaluation.

## Learning to Rank

A pointwise LTR model (logistic regression trained by gradient descent, L2 regularization, early stopping) combines the scores of the 11 models as features.

- Train/test split at query level: 24 queries for training, 6 for testing
- Class imbalance (about 2.2% relevant pairs) handled by undersampling the majority class
- Test results: accuracy 0.94, F1 (relevant class) 0.36, recall 0.82, precision 0.23
- Most influential features: LSI (k = 100), VSM cosine, LM Laplace

## Evaluation metrics

Precision, Recall, F1, P@5, P@10, R-Precision, MRR, MAP, interpolated Precision–Recall curves, DCG@20, nDCG@20, and relative Gain (%).

## Results (average over 30 queries)

| Model | MAP | MRR | P@5 | P@10 | R-Prec |
|-------|-----|-----|-----|------|--------|
| LSI (k = 100) | **0.6463** | 0.9278 | **0.8467** | **0.7667** | **0.6235** |
| VSM (Cosine) | 0.5139 | **0.9444** | 0.7333 | 0.6333 | 0.5117 |
| Extended BIR (no relevance) | 0.5024 | 0.8062 | 0.6333 | 0.6100 | 0.5329 |
| BM25 | 0.4777 | 0.8753 | 0.6600 | 0.5800 | 0.4964 |
| BIR (no relevance) | 0.4749 | 0.7854 | 0.6867 | 0.5833 | 0.4589 |
| LM Laplace | 0.4563 | 0.9043 | 0.6667 | 0.5600 | 0.4428 |
| LM Jelinek–Mercer | 0.3559 | 0.5818 | 0.4800 | 0.4333 | 0.3476 |
| LM Dirichlet | 0.3537 | 0.5957 | 0.4667 | 0.4200 | 0.3353 |
| Extended BIR (with relevance) | 0.1529 | 0.4098 | 0.2800 | 0.2333 | 0.1847 |
| BIR (with relevance) | 0.1056 | 0.2946 | 0.2067 | 0.1700 | 0.1254 |
| LM MLE | 0.0590 | 0.1167 | 0.0667 | 0.0633 | 0.0309 |

**Notes**

- Precision, Recall and F1 are constant across models because every model returns the whole collection. Use a score threshold or Top-K to make them discriminative.
- nDCG@20 is computed from the models' own scores, so it is 1 for most models and reflects internal consistency, not real ranking quality.

## Streamlit app

The interface lets you:

- pick a query and a retrieval model
- browse the top-K ranked documents (rank, id, score, excerpt)
- view per-query and global metrics
- compare all models in a table and in charts
- inspect standard and interpolated Precision–Recall curves
- switch between light and dark themes

### Run locally

```bash
git clone https://github.com/sarahmoussaoui/Information-Retrival-Project.git
cd Information-Retrival-Project

pip install -r requirements.txt
streamlit run app.py   # adjust to the actual entry file name
```

### Docker

```bash
docker build -t ir-app .
docker run -p 8080:8080 ir-app
```

The deployed version is built with Google Cloud Build and served on Google Cloud Run.

## Repository structure

```
.
├── FinalReport/                 # Project report (PDF)
├── SourceCode/
│   └── evaluation_results/      # Per-query and per-model JSON results
│       └── evaluation_results_dcg_ndcg_gain/
│           └── comparison_report.json
├── ...                          # Add the remaining folders and files here
└── README.md
```

Per-query gain results (nDCG and DCG) are in `comparison_report.json`.

