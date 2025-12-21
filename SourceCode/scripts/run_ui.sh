#!/usr/bin/env bash
# Launch the Streamlit UI

set -euo pipefail

echo "Launching Information Retrieval System UI..."
echo "=========================================="
echo ""

# Navigate to SourceCode directory
cd "$(dirname "$0")/.."

# Run Streamlit
streamlit run src/ui/streamlit_app.py
