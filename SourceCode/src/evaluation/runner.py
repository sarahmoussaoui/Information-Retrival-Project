"""Run all models on the query set and persist outputs."""

import json
import os
import sys
from pathlib import Path
import pandas as pd

def load_results_json(results_path):
    """Load a results JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_qrels(qrels_path):
    """Load qrels (relevance judgments) JSON file."""
    with open(qrels_path, 'r', encoding='utf-8') as f:
        qrels = json.load(f)
    # Convert string keys to int for consistency
    return {int(k): v for k, v in qrels.items()}


def run_all(models, queries, index, output_dir="outputs"):
    raise NotImplementedError
