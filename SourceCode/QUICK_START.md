# 🚀 Quick Start Guide - IR System UI

## Step-by-Step Instructions

### 1. Install Dependencies

Open a terminal in the `SourceCode` directory and run:

```bash
pip install -r requirements.txt
```

This will install:
- streamlit (Web UI framework)
- plotly (Interactive charts)
- pandas (Data manipulation)
- numpy (Numerical operations)
- And other required packages

### 2. Verify Data Files

Make sure you have the following directories with data:
- ✅ `data/processed/parse_preprocess/` - Contains queries and documents
- ✅ `Results/` - Contains model results (.json files)
- ✅ `Results/curves/` - Contains PR curve images
- ✅ `evaluation_results/evaluation_results_dcg_ndcg_gain/` - Contains metrics

### 3. Launch the UI

**Option A - Windows (Recommended):**
```bash
cd scripts
run_ui.bat
```

**Option B - Direct Command:**
```bash
cd SourceCode
streamlit run src/ui/streamlit_app.py
```

### 4. Use the UI

1. **Select a Query** from the sidebar (Start with Query 1)
2. **Select a Model** (Try BM25 first)
3. **Explore 4 Tabs:**
   - 📄 **Results**: See ranked documents
   - 📊 **Metrics**: View DCG, NDCG scores
   - 🔄 **Model Comparison**: Compare all models
   - 📈 **PR Curves**: See Precision-Recall curves

### 5. Take Screenshots

For your project demonstration, capture:
- Query 1 results with BM25 model
- Metrics tab showing DCG/NDCG
- Model comparison chart
- PR curves (both standard and interpolated)

## 🎯 Example Workflow

```
1. Open terminal
2. cd Information-Retrival-Project/SourceCode
3. pip install -r requirements.txt
4. cd scripts
5. run_ui.bat
6. Browser opens automatically at http://localhost:8501
7. Select Query 1, Model: BM25
8. Explore all 4 tabs
9. Take screenshots
```

## ⚡ Troubleshooting

**Problem: "Module not found"**
```bash
pip install streamlit plotly pandas
```

**Problem: "File not found"**
- Make sure you're in the SourceCode directory
- Check that Results/ and data/ folders exist

**Problem: "Port already in use"**
```bash
streamlit run src/ui/streamlit_app.py --server.port 8502
```

## 📌 Important Notes

- The UI runs on `http://localhost:8501` by default
- Use Ctrl+C in terminal to stop the UI
- The UI is read-only - it won't modify your data
- All visualizations are generated in real-time

## 🎨 Features You'll See

✨ **Clean Interface** - Modern, professional design
✨ **Interactive Charts** - Hover to see details
✨ **Responsive Layout** - Works on different screen sizes
✨ **Real-time Updates** - Changes reflect immediately
✨ **Complete Documentation** - Built-in help and tooltips

---

**Ready to impress! 🌟**
