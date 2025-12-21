# Information Retrieval System - User Interface

## 📋 Overview

This Streamlit-based User Interface provides an interactive way to explore and evaluate the Information Retrieval system results.

## ✨ Features

### 1. **Query Selection**
- Browse and select from 30 available queries
- View the preprocessed query text
- Navigate easily between different queries

### 2. **Ranked Results Display**
- View top-K ranked documents for any query
- See document IDs, scores, and content previews
- Adjustable number of results (5-50)
- Clear visual ranking with badges

### 3. **Evaluation Metrics Visualization**
- **DCG@20**: Discounted Cumulative Gain at rank 20
- **NDCG@20**: Normalized Discounted Cumulative Gain at rank 20
- **Precision@10**: Precision at rank 10 (if available)
- **Recall@10**: Recall at rank 10 (if available)
- Score distribution charts for top 20 documents

### 4. **Model Comparison**
- Compare all retrieval models side-by-side
- Interactive bar charts for visual comparison
- Sortable comparison tables

### 5. **Precision-Recall Curves**
- Standard Precision-Recall curves
- Interpolated Precision-Recall curves
- Available for all implemented models

## 🎯 Supported Models

The UI supports the following retrieval models:
- **BM25**: Best Match 25
- **VSM_Cosine**: Vector Space Model with Cosine similarity
- **LM_MLE**: Language Model with Maximum Likelihood Estimation
- **LM_Laplace**: Language Model with Laplace smoothing
- **LM_JelinekMercer**: Language Model with Jelinek-Mercer smoothing
- **LM_Dirichlet**: Language Model with Dirichlet smoothing
- **LSI_k100**: Latent Semantic Indexing (k=100)
- **BIR_no_relevance**: Binary Independent Retrieval (without relevance feedback)
- **BIR_with_relevance**: Binary Independent Retrieval (with relevance feedback)
- **ExtendedBIR_no_relevance**: Extended BIR (without relevance feedback)
- **ExtendedBIR_with_relevance**: Extended BIR (with relevance feedback)

## 🚀 How to Run

### Prerequisites

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- streamlit
- plotly
- pandas
- numpy

### Running the UI

#### On Windows:
```bash
cd SourceCode/scripts
run_ui.bat
```

#### On Linux/Mac:
```bash
cd SourceCode/scripts
bash run_ui.sh
```

#### Manual Launch:
```bash
cd SourceCode
streamlit run src/ui/streamlit_app.py
```

The UI will automatically open in your default web browser at `http://localhost:8501`

## 📊 UI Structure

### Sidebar (Left)
- **Query Selection**: Choose which query to explore
- **Model Selection**: Select the retrieval model
- **Display Options**: Adjust number of results to show

### Main Content (4 Tabs)

#### Tab 1: 📄 Results
- Displays ranked documents for the selected query and model
- Shows document ID, relevance score, and content preview
- Results are numbered and color-coded

#### Tab 2: 📊 Metrics
- Displays evaluation metrics (DCG, NDCG, Precision, Recall)
- Shows score distribution chart
- Provides quick performance insights

#### Tab 3: 🔄 Model Comparison
- Compares all models for the selected query
- Interactive bar charts
- Sortable comparison table

#### Tab 4: 📈 PR Curves
- Standard Precision-Recall curve
- Interpolated Precision-Recall curve
- Visual performance evaluation

## 💡 Usage Tips

1. **Start with Query 1**: It's a good example to demonstrate the system
2. **Compare Models**: Use the "Model Comparison" tab to see which model performs best for a specific query
3. **Adjust Top-K**: Change the number of results to see more or fewer documents
4. **Explore Different Queries**: Each query has different characteristics and model performance

## 📸 Screenshots

When demonstrating your project, capture screenshots of:
1. Query 1 results with a specific model (e.g., BM25)
2. Evaluation metrics display
3. Model comparison chart
4. Precision-Recall curves

## 🛠️ Troubleshooting

### "No results found for this query"
- Ensure all model results are generated in `Results/` directory
- Run the evaluation pipeline first

### "PR curves not found"
- Check that `Results/curves/` contains the PR curve images
- Generate curves using `scripts/make_plots.py`

### Module import errors
- Make sure you're running from the `SourceCode` directory
- Verify all dependencies are installed

## 📁 Required Data Files

The UI requires the following files:
```
SourceCode/
├── data/processed/parse_preprocess/
│   ├── queries_processed.json
│   └── docs_processed.json
├── Results/
│   ├── [MODEL_NAME].json (for each model)
│   └── curves/
│       ├── PR_Curve_*.png
│       └── Interpolated_PR_Curve_*.png
└── evaluation_results/evaluation_results_dcg_ndcg_gain/
    ├── [MODEL_NAME]_metrics.json
    └── all_models_comparison.json
```

## 🎨 Customization

You can customize the UI by modifying:
- **Colors**: Edit the CSS in `streamlit_app.py`
- **Metrics**: Add more metrics in `display_metrics()` function
- **Charts**: Modify Plotly charts for different visualizations
- **Layout**: Adjust column sizes and tab organization

## 📝 Notes

- The UI is read-only and won't modify any data files
- All visualizations are generated dynamically
- The system supports up to 30 queries by default
- Performance is optimized for standard result sets

## 🤝 Contributing

To add new features to the UI:
1. Edit `src/ui/streamlit_app.py` for main UI logic
2. Add helper functions to `src/ui/components.py`
3. Test with different queries and models
4. Update this README with new features

---

**Enjoy exploring your Information Retrieval System! 🚀**
