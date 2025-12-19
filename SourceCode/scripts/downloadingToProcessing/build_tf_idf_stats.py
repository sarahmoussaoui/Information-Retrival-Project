"""Compute TF/TF-max/doc-freq/n-docs and normalized variants from processed tokens.

Inputs (required):
- data/processed/parse_preprocess/docs_processed.json : {doc_id: [tokens]}
- data/processed/parse_preprocess/queries_processed.json : {query_id: [tokens]} (not used for doc stats but can be extended)

Outputs (each saved as a separate JSON file under data/processed/build_tf_idf_stats/):
    - n_docs.json                 : {"n_docs": int}
    - doc_tf.json                 : {doc_id: {term: tf}}
    - doc_tf_max.json             : {doc_id: tf_max}
    - doc_tf_norm.json            : {doc_id: {term: tf_norm}}           # tf_norm = tf / tf_max for that doc (0 if tf_max=0)
    - doc_freq.json               : {term: df}
    - collection_tf.json          : {term: total_tf_across_all_docs}
    - collection_tf_norm.json     : {term: tf_norm_collection}          # tf_norm_collection = tf_total / max(tf_total) over terms
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from indexing.weighting import tf_norm, tf_idf


def load_tokens(path: Path) -> dict[int, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # JSON keys are strings; convert to int
    return {int(k): v for k, v in raw.items()}


def compute_stats(doc_tokens: dict[int, list[str]]):
    n_docs = len(doc_tokens)

    doc_tf = {}
    doc_tf_max = {}
    doc_tf_norm = {}
    doc_freq = Counter()
    collection_tf = Counter()

    # Per-document TF and tf_max
    for doc_id, tokens in doc_tokens.items():
        counts = Counter(tokens)
        doc_tf[doc_id] = dict(counts)
        collection_tf.update(counts)
        tf_max = max(counts.values()) if counts else 0
        doc_tf_max[doc_id] = tf_max
        # Normalized TF per document
        if tf_max > 0:
            doc_tf_norm[doc_id] = {term: tf_norm(tf, tf_max) for term, tf in counts.items()}
        else:
            doc_tf_norm[doc_id] = {term: 0.0 for term in counts.keys()}
        # Update document frequency per term (presence in doc)
        for term in counts:
            doc_freq[term] += 1

    # Normalized collection TF: divide by max total frequency across all terms
    if collection_tf:
        max_collection_tf = max(collection_tf.values())
        collection_tf_norm = {term: tf_norm(total_tf, max_collection_tf) for term, total_tf in collection_tf.items()}
    else:
        collection_tf_norm = {}

    return {
        "n_docs": n_docs,
        "doc_tf": doc_tf,
        "doc_tf_max": doc_tf_max,
        "doc_tf_norm": doc_tf_norm,
        "doc_freq": dict(doc_freq),
        "collection_tf": dict(collection_tf),
        "collection_tf_norm": collection_tf_norm,
    }


def main():
    base = Path(__file__).parent.parent.parent
    processed_root = base / "data" / "processed"
    docs_dir = processed_root / "parse_preprocess"
    out_dir = processed_root / "build_tf_idf_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs_path = docs_dir / "docs_processed.json"
    if not docs_path.exists():
        raise FileNotFoundError(f"Missing {docs_path}")

    # Load documents
    doc_tokens = load_tokens(docs_path)

    stats = compute_stats(doc_tokens)

    # Save stats to separate files
    def save_json(path: Path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    save_json(out_dir / "n_docs.json", {"n_docs": stats["n_docs"]})
    save_json(out_dir / "doc_tf.json", stats["doc_tf"])
    save_json(out_dir / "doc_tf_max.json", stats["doc_tf_max"])
    save_json(out_dir / "doc_tf_norm.json", stats["doc_tf_norm"])
    save_json(out_dir / "doc_freq.json", stats["doc_freq"])
    save_json(out_dir / "collection_tf.json", stats["collection_tf"])
    save_json(out_dir / "collection_tf_norm.json", stats["collection_tf_norm"])

    # Optional: print a small sample
    sample_doc = next(iter(doc_tokens)) if doc_tokens else None
    if sample_doc is not None:
        tf_max = stats["doc_tf_max"][sample_doc]
        tf_sample = list(stats["doc_tf"][sample_doc].items())[:5]
        print(f"n_docs={stats['n_docs']}")
        print(f"sample doc={sample_doc}, tf_max={tf_max}, tf_sample={tf_sample}")
        term = tf_sample[0][0] if tf_sample else None
        if term:
            df = stats["doc_freq"].get(term, 0)
            tf_raw = stats["doc_tf"][sample_doc][term]
            weight = tf_idf(tf_raw, tf_max, df, stats["n_docs"])
            print(f"example weight(term='{term}')={weight:.4f}")


if __name__ == "__main__":
    main()
