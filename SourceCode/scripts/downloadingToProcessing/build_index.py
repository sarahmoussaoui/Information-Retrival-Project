"""CLI to build Document–Term Matrix and Inverted Index."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import save_npz

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from indexing.doc_term_matrix import build_matrix, build_binary_matrix
from indexing.inverted_index import build_inverted_index


def load_processed_docs(docs_dir: Path) -> dict[int, list[str]]:
    docs_path = docs_dir / "docs_processed.json"
    if not docs_path.exists():
        raise FileNotFoundError(f"Missing {docs_path}")
    with open(docs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # keys are strings in JSON; convert to int
    return {int(k): v for k, v in data.items()}


def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def main():
    base = Path(__file__).parent.parent.parent
    processed_root = base / "data" / "processed"
    docs_dir = processed_root / "parse_preprocess"
    out_dir = processed_root / "build_index"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_processed_docs(docs_dir)

    # Precompute doc lengths (number of tokens per document) and average length
    doc_lengths = {doc_id: len(tokens) for doc_id, tokens in docs.items()}
    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0.0

    # Build Document–Term Matrix
    matrix, vocab, doc_index = build_matrix(docs)
    save_npz(out_dir / "doc_term_matrix.npz", matrix)
    # Build Binary Document–Term Matrix (presence/absence)
    bin_matrix = build_binary_matrix(docs, vocab, doc_index)
    save_npz(out_dir / "doc_term_matrix_binary.npz", bin_matrix)
    save_json(out_dir / "vocab.json", vocab)
    save_json(out_dir / "doc_index.json", doc_index)
    save_json(out_dir / "doc_lengths.json", doc_lengths)
    save_json(out_dir / "avg_doc_length.json", {"avg_doc_length": avg_doc_length})

    # Build Inverted Index 
    inverted = build_inverted_index(docs)
    save_json(out_dir / "inverted_index.json", inverted)

    print("==================================================")
    print("Indexes built and saved to data/processed")
    print(f"Documents: {len(docs)}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Matrix shape: {matrix.shape}")
    print("Files written to data/processed/build_index: doc_term_matrix.npz, doc_term_matrix_binary.npz, vocab.json, doc_index.json, doc_lengths.json, avg_doc_length.json, inverted_index.json, doc_term_counts.json")
    print("==================================================")


if __name__ == "__main__":
    main()
