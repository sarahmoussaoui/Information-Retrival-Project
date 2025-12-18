"""Test script to verify MEDLINE parsing."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dataset.parser import parse_documents, parse_queries, parse_qrels
from dataset.preprocessing import preprocess_mapping


def main():
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    processed_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "parse_preprocess"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Test parse_documents
    print("=" * 60)
    print("Testing parse_documents (MED.ALL)")
    print("=" * 60)
    with open(data_dir / "MED.ALL", "r", encoding="utf-8") as f:
        docs = parse_documents(f.read())

    processed_docs = preprocess_mapping(docs)
    with open(processed_dir / "docs_processed.json", "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, ensure_ascii=False)
    
    print(f"Total documents: {len(docs)}")
    print(f"\nFirst 3 document IDs: {sorted(docs.keys())[:3]}")
    print(f"\nSample document (ID=1):")
    print(f"  Raw: {docs[1][:200]}...")
    print(f"  Preprocessed: {processed_docs[1][:20]}...")
    
    # Test parse_queries
    print("\n" + "=" * 60)
    print("Testing parse_queries (MED.QRY)")
    print("=" * 60)
    with open(data_dir / "MED.QRY", "r", encoding="utf-8") as f:
        queries = parse_queries(f.read())

    processed_queries = preprocess_mapping(queries)
    with open(processed_dir / "queries_processed.json", "w", encoding="utf-8") as f:
        json.dump(processed_queries, f, ensure_ascii=False)
    
    print(f"Total queries: {len(queries)}")
    print(f"\nFirst 3 query IDs: {sorted(queries.keys())[:3]}")
    print(f"\nSample query (ID=1):")
    print(f"  Raw: {queries[1]}")
    print(f"  Preprocessed: {processed_queries[1]}")
    
    # Test parse_qrels
    print("\n" + "=" * 60)
    print("Testing parse_qrels (MED.REL)")
    print("=" * 60)
    with open(data_dir / "MED.REL", "r", encoding="utf-8") as f:
        qrels = parse_qrels(f.read())

    # Save qrels as lists for JSON compatibility
    qrels_as_lists = {qid: sorted(list(docs)) for qid, docs in qrels.items()}
    with open(processed_dir / "qrels.json", "w", encoding="utf-8") as f:
        json.dump(qrels_as_lists, f, ensure_ascii=False)
    
    print(f"Total queries with relevance judgments: {len(qrels)}")
    print(f"\nFirst 3 queries: {sorted(qrels.keys())[:3]}")
    print(f"\nQuery 1 relevant docs: {sorted(qrels[1])[:10]}... ({len(qrels[1])} total)")
    print(f"Query 2 relevant docs: {sorted(qrels[2])[:10]}... ({len(qrels[2])} total)")
    print(f"Query 5 relevant docs: {sorted(qrels[5])[:10]}... ({len(qrels[5])} total)")
    
    print("\n" + "=" * 60)
    print("✓ All parsing tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
