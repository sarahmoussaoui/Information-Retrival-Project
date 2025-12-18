`MED.REL.OLD` is **ignored**.

### **MED.ALL**

**Purpose:** document collection (indexing)

this file is used to:

* Parse **1,033 documents**
* Apply preprocessing:

  * tokenization (same regex as previous labs)
  * stopword removal
  * Porter stemming
* Compute:

  * TF
  * TF-IDF
* Build:

  * Document–Term Matrix
  * Inverted Index

This is the **only file indexed**.

---

### ✅ **MED.QRY** 

**Purpose:** queries

This file is used to:

* Parse the **30 queries**
* Apply **the same preprocessing pipeline** as documents
* Feed queries to all retrieval models

Queries are **NOT indexed**.

---

### ✅ **MED.REL** — USE IT

**Purpose:** relevance judgments (ground truth)

This file is used to:

* Know which documents are relevant to each query
* Compute **all evaluation metrics**
* Feed relevance-based models:

  * Classic BIR (with relevance)
  * Extended BIR (with relevance)

This file is **never preprocessed**.

---

### ❌ **MED.REL.OLD** — NOT USED

**Purpose:** obsolete / legacy relevance file

You:

* Do **not** load it
* Do **not** reference it in code
* Do **not** mention it in experiments

---

## One-line summary

> MED.ALL is used for indexing and term weighting, MED.QRY for query processing, MED.REL for evaluation and relevance-based models, while MED.REL.OLD is ignored as an obsolete relevance file.