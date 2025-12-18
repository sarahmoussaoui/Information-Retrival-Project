"""Compute TF/TF-max/doc-freq/n-docs from processed tokens.

Inputs (required):
- data/processed/parse_preprocess/docs_processed.json : {doc_id: [tokens]}
- data/processed/parse_preprocess/queries_processed.json : {query_id: [tokens]} (not used for doc stats but can be extended)

Outputs:
- data/processed/tf_idf_stats.json with:
    {
      "n_docs": int,
      "doc_tf": {doc_id: {term: tf}},
      "doc_tf_max": {doc_id: tf_max},
    "doc_freq": {term: df}
    "collection_tf": {term: total_tf_across_all_docs}
    }
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
    doc_freq = Counter()
    collection_tf = Counter()

    # Per-document TF and tf_max
    for doc_id, tokens in doc_tokens.items():
        counts = Counter(tokens)
        doc_tf[doc_id] = dict(counts)
        collection_tf.update(counts)
        tf_max = max(counts.values()) if counts else 0
        doc_tf_max[doc_id] = tf_max
        # Update document frequency per term (presence in doc)
        for term in counts:
            doc_freq[term] += 1

    return {
        "n_docs": n_docs,
        "doc_tf": doc_tf,
        "doc_tf_max": doc_tf_max,
        "doc_freq": dict(doc_freq),
        "collection_tf": dict(collection_tf),
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

    # Save stats
    out_path = out_dir / "tf_idf_stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False)

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
