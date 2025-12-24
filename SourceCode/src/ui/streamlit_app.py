"""Streamlit UI for Information Retrieval System."""

import streamlit as st
import json
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from components import (
    load_queries,
    load_documents,
    load_results,
    load_metrics,
    get_available_models,
    get_query_ids
)

# Page configuration
st.set_page_config(
    page_title="IR System - Query Search & Evaluation",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Theme CSS
def inject_theme_css(mode):
    if mode == "Dark":
        st.markdown("""
        <style>
            /* Global Styles */
            body, .stApp { 
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important; 
                color: #e0e0e0 !important;
                font-family: 'Inter', 'Segoe UI', sans-serif;
            }
            
            /* Header */
            .main-header { 
                font-size: 3rem; 
                font-weight: 700;
                background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 50%, #0288d1 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center; 
                margin-bottom: 1.5rem;
                text-shadow: 0 0 30px rgba(79, 195, 247, 0.3);
            }
            
            .subtitle {
                text-align: center;
                color: #90caf9;
                font-size: 1.1rem;
                margin-bottom: 2rem;
                font-weight: 300;
            }
            
            /* Metric Cards */
            .metric-card { 
                background: linear-gradient(135deg, #23272f 0%, #2d3748 100%);
                color: #e0e0e0;
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 8px 16px rgba(0,0,0,0.3);
                border: 1px solid rgba(79, 195, 247, 0.2);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            .metric-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 12px 24px rgba(79, 195, 247, 0.3);
            }
            
            /* Result Items */
            .result-item { 
                background: linear-gradient(135deg, #23272f 0%, #2d3748 100%);
                color: #e0e0e0;
                padding: 1.5rem;
                border-left: 5px solid #4fc3f7;
                border-radius: 8px;
                margin: 1rem 0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
            }
            
            .result-item:hover {
                transform: translateX(8px);
                box-shadow: 0 6px 20px rgba(79, 195, 247, 0.3);
                border-left-width: 6px;
            }
            
            /* Rank Badge */
            .rank-badge { 
                background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%);
                color: #0d1117;
                padding: 0.6rem 1rem;
                border-radius: 50px;
                font-weight: 700;
                font-size: 1.1rem;
                box-shadow: 0 4px 12px rgba(79, 195, 247, 0.4);
                display: inline-block;
                text-align: center;
            }
            
            /* Sidebar */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
                border-right: 1px solid rgba(79, 195, 247, 0.2);
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background-color: rgba(79, 195, 247, 0.1);
                border-radius: 8px 8px 0 0;
                padding: 12px 24px;
                color: #90caf9;
                font-weight: 600;
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%);
                color: #0d1117;
            }
            
            /* Footer */
            .footer {
                text-align: center;
                padding: 2rem;
                margin-top: 4rem;
                color: #90caf9;
                border-top: 1px solid rgba(79, 195, 247, 0.2);
                font-size: 0.9rem;
            }
            
            /* Animations */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .animate-fade {
                animation: fadeIn 0.5s ease-out;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            /* Global Styles - Light Mode */
            body, .stApp { 
                background: linear-gradient(135deg, #fafbfc 0%, #f0f4f8 100%) !important;
                color: #1e293b !important;
                font-family: 'Inter', 'Segoe UI', Tahoma, sans-serif;
            }
            
            /* Header */
            .main-header { 
                font-size: 3rem; 
                font-weight: 800;
                background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #06b6d4 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center; 
                margin-bottom: 1.5rem;
                letter-spacing: -1.5px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            .subtitle {
                text-align: center;
                color: #64748b;
                font-size: 1.15rem;
                margin-bottom: 2rem;
                font-weight: 500;
                letter-spacing: 0.3px;
            }
            
            /* Metric Cards */
            .metric-card { 
                background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%);
                color: #0f172a;
                padding: 1.5rem;
                border-radius: 16px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 10px 15px rgba(0, 0, 0, 0.03);
                border: 1px solid #e2e8f0;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            .metric-card:hover {
                transform: translateY(-6px);
                box-shadow: 0 12px 24px rgba(37, 99, 235, 0.15), 0 4px 8px rgba(0, 0, 0, 0.05);
                border-color: #bfdbfe;
            }
            
            /* Result Items */
            .result-item { 
                background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                padding: 1.8rem;
                border-left: 5px solid #3b82f6;
                border-radius: 12px;
                margin: 1.2rem 0;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05), 0 4px 12px rgba(0, 0, 0, 0.04);
                border-right: 1px solid #e2e8f0;
                border-top: 1px solid #e2e8f0;
                border-bottom: 1px solid #e2e8f0;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            .result-item:hover {
                transform: translateX(10px);
                box-shadow: 0 8px 16px rgba(37, 99, 235, 0.18), 0 4px 8px rgba(0, 0, 0, 0.06);
                border-left-width: 6px;
                border-left-color: #2563eb;
                background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
            }
            
            /* Text in result items */
            .result-item strong {
                color: #0f172a;
                font-weight: 700;
            }
            
            .result-item p {
                color: #334155;
                line-height: 1.7;
            }
            
            /* Rank Badge */
            .rank-badge { 
                background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
                color: white;
                padding: 0.7rem 1.2rem;
                border-radius: 50px;
                font-weight: 800;
                font-size: 1.1rem;
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.35);
                display: inline-block;
                text-align: center;
                border: 2px solid rgba(255, 255, 255, 0.3);
            }
            
            /* Sidebar */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
                border-right: 2px solid #e2e8f0;
                box-shadow: 2px 0 8px rgba(0, 0, 0, 0.03);
            }
            
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: #0f172a;
                font-weight: 700;
            }
            
            [data-testid="stSidebar"] label {
                color: #475569;
                font-weight: 600;
            }
            
            [data-testid="stSidebar"] .stSelectbox,
            [data-testid="stSidebar"] .stSlider,
            [data-testid="stSidebar"] .stRadio {
                background-color: rgba(248, 250, 252, 0.5);
                padding: 0.5rem;
                border-radius: 8px;
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 10px;
                background-color: transparent;
                padding: 0.5rem 0;
            }
            
            .stTabs [data-baseweb="tab"] {
                background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
                border-radius: 10px 10px 0 0;
                padding: 14px 28px;
                color: #475569;
                font-weight: 700;
                border: 2px solid #cbd5e1;
                border-bottom: none;
                transition: all 0.3s ease;
                text-transform: uppercase;
                font-size: 0.9rem;
                letter-spacing: 0.5px;
            }
            
            .stTabs [data-baseweb="tab"]:hover {
                background: linear-gradient(135deg, #e0e7ff 0%, #dbeafe 100%);
                color: #3b82f6;
                transform: translateY(-2px);
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
                color: white;
                border-color: #2563eb;
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
            }
            
            /* Footer */
            .footer {
                text-align: center;
                padding: 2.5rem;
                margin-top: 4rem;
                color: #64748b;
                border-top: 2px solid #e2e8f0;
                font-size: 0.95rem;
                background: linear-gradient(135deg, #fafbfc 0%, #f8fafc 100%);
                border-radius: 12px 12px 0 0;
            }
            
            .footer strong {
                color: #1e293b;
                font-weight: 700;
            }
            
            /* Info boxes */
            .stAlert {
                border-radius: 10px;
                border-left: 5px solid #3b82f6;
                background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
                box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
                color: #1e40af;
            }
            
            /* Success/Warning boxes */
            .stSuccess {
                background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                border-left: 5px solid #22c55e;
                color: #15803d;
            }
            
            .stWarning {
                background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
                border-left: 5px solid #f59e0b;
                color: #b45309;
            }
            
            /* Animations */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .animate-fade {
                animation: fadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            /* DataFrames */
            .dataframe {
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
                border: 1px solid #e2e8f0;
            }
            
            .dataframe thead tr {
                background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
                color: white;
                font-weight: 700;
            }
            
            .dataframe tbody tr {
                color: #334155;
            }
            
            .dataframe tbody tr:hover {
                background-color: #f1f5f9;
                transition: background-color 0.2s ease;
            }
            
            /* Buttons */
            .stButton > button {
                border-radius: 10px;
                font-weight: 700;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
                color: white;
                border: none;
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .stButton > button:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 20px rgba(59, 130, 246, 0.35);
                background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
            }
            
            /* Metrics */
            [data-testid="stMetricValue"] {
                font-size: 2rem;
                font-weight: 800;
                color: #2563eb;
            }
            
            [data-testid="stMetricLabel"] {
                color: #64748b;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-size: 0.85rem;
            }
            
            /* Sliders */
            .stSlider {
                padding: 1rem 0;
            }
            
            /* Headings */
            h1 {
                color: #0f172a;
                font-weight: 800;
            }
            
            h2 {
                color: #1e293b;
                font-weight: 700;
            }
            
            h3 {
                color: #334155;
                font-weight: 600;
            }
            
            /* Paragraphs */
            p {
                color: #475569;
                line-height: 1.7;
            }
            
            /* Links */
            a {
                color: #3b82f6;
                text-decoration: none;
                font-weight: 600;
                transition: color 0.2s ease;
            }
            
            a:hover {
                color: #2563eb;
                text-decoration: underline;
            }
            
            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 10px;
                height: 10px;
            }
            
            ::-webkit-scrollbar-track {
                background: #f1f5f9;
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
            }
        </style>
        """, unsafe_allow_html=True)



def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">Information Retrieval System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced search and evaluation platform for document retrieval</p>', unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar():
    """Render the sidebar with query and model selection."""
    st.sidebar.title("Configuration")
    st.sidebar.markdown("---")
    theme_mode = st.sidebar.radio("Theme Mode", ["Light", "Dark"], index=1)
    st.session_state["theme_mode"] = theme_mode
    st.sidebar.markdown("---")
    queries = load_queries()
    query_ids = get_query_ids(queries)
    
    # Query selection
    st.sidebar.subheader("Select Query")
    selected_query = st.sidebar.selectbox(
        "Query ID",
        options=query_ids,
        format_func=lambda x: f"Query {x}"
    )
    
    # Display selected query text
    if selected_query and queries:
        query_text = " ".join(queries.get(str(selected_query), []))
        st.sidebar.info(f"**Query Text:**\n{query_text}")
    
    st.sidebar.markdown("---")
    
    # Model selection
    st.sidebar.subheader("Select Model")
    available_models = get_available_models()
    # Always include 'Avancée' as an option
    models_with_avancee = list(available_models)
    if "Avancée" not in models_with_avancee:
        models_with_avancee.append("Avancée")
    selected_model = st.sidebar.selectbox(
        "Retrieval Model",
        options=models_with_avancee
    )
    
    st.sidebar.markdown("---")
    
    # Top-K results
    st.sidebar.subheader("Display Options")
    top_k = st.sidebar.slider("Number of results to display", 5, 50, 20)
    
    # Add info section
    st.sidebar.markdown("---")
    with st.sidebar.expander("About"):
        st.markdown("""
        **IR System Features:**
        - Multiple retrieval models
        - Comprehensive metrics
        - Visual comparisons
        - Theme customization
        """)
    
    return selected_query, selected_model, top_k


def display_ranked_results(query_id, model_name, documents, results, top_k=20):
    """Display ranked search results."""
    st.header(f"Ranked Results for Query {query_id} - {model_name}")
    
    query_results = results.get("queries", {}).get(str(query_id), [])
    
    if not query_results:
        st.warning("No results found for this query.")
        return
    
    # Display result count
    st.info(f"Showing top {min(top_k, len(query_results))} of {len(query_results)} results")
    
    # Display top-k results
    for idx, result in enumerate(query_results[:top_k], 1):
        doc_id = result["doc_id"]
        score = result["score"]
        
        # Get document text (if available)
        doc_text = documents.get(str(doc_id), "Document text not available")
        if isinstance(doc_text, list):
            doc_text = " ".join(doc_text)
        
        # Truncate long text
        doc_preview = doc_text[:300] + "..." if len(doc_text) > 300 else doc_text
        
        with st.container():
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown(f'<div class="rank-badge">#{idx}</div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="result-item animate-fade">', unsafe_allow_html=True)
                st.markdown(f"**Document ID:** {doc_id} | **Score:** {score:.4f}")
                st.markdown(f"**Content:** {doc_preview}")
                st.markdown('</div>', unsafe_allow_html=True)


def display_metrics(query_id, model_name, metrics_data):
    """Display evaluation metrics for selected query and model."""
    st.header("Evaluation Metrics")
    
    query_metrics = metrics_data.get("query_metrics", {}).get(str(query_id), {})
    
    if not query_metrics:
        st.warning("No metrics available for this query.")
        return
    
    # Display all available metrics except model_scores
    metric_keys = [k for k in query_metrics.keys() if k != "model_scores"]
    if metric_keys:
        st.subheader("Key Performance Indicators")
        cols = st.columns(min(4, len(metric_keys)))
        for idx, key in enumerate(metric_keys):
            value = query_metrics[key]
            # Format floats nicely
            if isinstance(value, float):
                value = f"{value:.4f}"
            # Add delta for context (placeholder - could be calculated from avg)
            cols[idx % len(cols)].metric(key.upper(), value)

    # Display score distribution
    if "model_scores" in query_metrics:
        scores = query_metrics["model_scores"][:20]
        fig = go.Figure(data=[
            go.Bar(x=list(range(1, len(scores)+1)), y=scores, marker_color='#1f77b4')
        ])
        fig.update_layout(
            title="Score Distribution (Top 20 Documents)",
            xaxis_title="Rank",
            yaxis_title="Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


def display_all_models_comparison(query_id):
    """Display comparison of all models for a specific query."""
    st.header("Model Comparison")
    
    # Load standard comparison data
    comparison_path = Path("evaluation_results/evaluation_results_dcg_ndcg_gain/all_models_comparison.json")
    avancee_path = Path("evaluation_results/evaluation_results_PRCurve_MAP_P@K_RR/evaluation_Avancée_results.json")
    basic_metrics_path = Path("evaluation_results/basic_metrics/comparison_report.json")
    
    if not comparison_path.exists():
        st.warning("Comparison data not available.")
        return
    
    with open(comparison_path, 'r') as f:
        all_models_data = json.load(f)
    
    # Load advanced metrics if available
    advanced_data = {}
    if avancee_path.exists():
        with open(avancee_path, 'r') as f:
            advanced_data = json.load(f)
    
    # Load basic metrics if available
    basic_data = {}
    if basic_metrics_path.exists():
        with open(basic_metrics_path, 'r') as f:
            basic_data = json.load(f)
    
    comparison_data = []
    for model_name, model_data in all_models_data.items():
        query_metrics = model_data.get("query_metrics", {}).get(str(query_id), {})
        if query_metrics:
            row = {
                "Model": model_name,
                "DCG@20": query_metrics.get("dcg@20", 0),
                "NDCG@20": query_metrics.get("ndcg@20", 0)
            }
            
            # Add advanced metrics if available
            if model_name in advanced_data:
                adv_query_metrics = advanced_data[model_name].get("PerQuery", {}).get(str(query_id), {})
                if adv_query_metrics:
                    row["MAP"] = adv_query_metrics.get("MAP", 0)
                    row["P@5"] = adv_query_metrics.get("P@5", 0)
                    row["P@10"] = adv_query_metrics.get("P@10", 0)
                    row["MRR"] = adv_query_metrics.get("MRR", 0)
            
            # Add basic metrics if available
            if model_name in basic_data:
                basic_query_metrics = basic_data[model_name].get("query_metrics", {}).get(str(query_id), {})
                if basic_query_metrics:
                    row["Precision"] = basic_query_metrics.get("precision", 0)
                    row["Recall"] = basic_query_metrics.get("recall", 0)
                    row["F1"] = basic_query_metrics.get("f1_score", 0)
                    row["R-Precision"] = basic_query_metrics.get("r_precision", 0)
            
            comparison_data.append(row)
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df = df.sort_values("NDCG@20", ascending=False)
        
        # Display table
        st.subheader("Complete Metrics Comparison")
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Create three columns for charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # DCG/NDCG metrics bar chart
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(name='DCG@20', x=df['Model'], y=df['DCG@20']))
            fig1.add_trace(go.Bar(name='NDCG@20', x=df['Model'], y=df['NDCG@20']))
            
            fig1.update_layout(
                title=f"Ranking Metrics - Query {query_id}",
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # MAP/MRR metrics bar chart (if available)
            if "MAP" in df.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(name='MAP', x=df['Model'], y=df['MAP']))
                fig2.add_trace(go.Bar(name='P@10', x=df['Model'], y=df['P@10']))
                fig2.add_trace(go.Bar(name='MRR', x=df['Model'], y=df['MRR']))
                
                fig2.update_layout(
                    title=f"Precision Metrics - Query {query_id}",
                    xaxis_title="Model",
                    yaxis_title="Score",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Extended metrics not available for comparison.")
        
        with col3:
            # Precision/Recall/F1 metrics (if available)
            if "Precision" in df.columns:
                fig3 = go.Figure()
                fig3.add_trace(go.Bar(name='Precision', x=df['Model'], y=df['Precision']))
                fig3.add_trace(go.Bar(name='Recall', x=df['Model'], y=df['Recall']))
                fig3.add_trace(go.Bar(name='F1', x=df['Model'], y=df['F1']))
                
                fig3.update_layout(
                    title=f"Evaluation Metrics - Query {query_id}",
                    xaxis_title="Model",
                    yaxis_title="Score",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Precision/Recall metrics not available for comparison.")


def display_pr_curves(model_name):
    """Display Precision-Recall curves."""
    st.header("Precision-Recall Curves")
    
    # Try both possible PR curve directories
    curve_dirs = [
        Path("evaluation_results/evaluation_results_PRCurve_MAP_P@K_RR/Curves"),
        Path("evaluation_results/evaluation_results_dcg_ndcg_gain/Curves")
    ]
    found_dir = None
    for d in curve_dirs:
        if d.exists():
            found_dir = d
            break
    if not found_dir:
        st.warning("PR curves directory not found in either location.")
        return

    # Map model names to file names
    model_file_map = {
        "BM25": "BM25",
        "VSM_Cosine": "VSMCosine",
        "LM_MLE": "LM_MLE",
        "LM_Laplace": "LM_Laplace",
        "LM_JelinekMercer": "LM_JelinekMercer",
        "LM_Dirichlet": "Dirichlet",
        "LSI_k100": "LSI_k100",
        "BIR_no_relevance": "BIR_no_relevance",
        "BIR_with_relevance": "BIR_with_relevance",
        "ExtendedBIR_no_relevance": "Extented_BIR_no_relevance",
        "ExtendedBIR_with_relevance": "Extented_BIR_with_relevance"
    }

    file_suffix = model_file_map.get(model_name, model_name)

    col1, col2 = st.columns(2)

    with col1:
        # Standard PR Curve
        pr_curve_path = found_dir / f"PR_Curve_{file_suffix}.png"
        if pr_curve_path.exists():
            st.subheader("Standard PR Curve")
            st.image(str(pr_curve_path), use_container_width=True)
        else:
            st.warning(f"PR curve not found: {pr_curve_path.name}")

    with col2:
        # Interpolated PR Curve
        interp_pr_path = found_dir / f"Interpolated_PR_Curve_{file_suffix}.png"
        if not interp_pr_path.exists():
            interp_pr_path = found_dir / f"interpolated_PR_Curve_{file_suffix}.png"

        if interp_pr_path.exists():
            st.subheader("Interpolated PR Curve")
            st.image(str(interp_pr_path), use_container_width=True)
        else:
            st.warning("Interpolated PR curve not found")


def main():
    """Main application."""
    # Sidebar with theme toggle
    selected_query, selected_model, top_k = render_sidebar()
    theme_mode = st.session_state.get('theme_mode', 'Dark')
    inject_theme_css(theme_mode)
    render_header()

    # Load data
    queries = load_queries()
    documents = load_documents()

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Results", "Metrics", "Model Comparison", "PR Curves"])

    with tab1:
        if selected_query and selected_model:
            results = load_results(selected_model)
            if results:
                display_ranked_results(selected_query, selected_model, documents, results, top_k)
            else:
                st.warning("No results available for this model.")
        else:
            st.info("Select a query and model from the sidebar to view results.")
    
    with tab2:
            if selected_query and selected_model:
                metrics = load_metrics(selected_model)
                if metrics:
                    display_metrics(selected_query, selected_model, metrics)

                # Advanced metrics section below standard metrics
                avancee_path = Path("evaluation_results/evaluation_results_PRCurve_MAP_P@K_RR/evaluation_Avancée_results.json")
                if avancee_path.exists():
                    with open(avancee_path, "r") as f:
                        avancee_data = json.load(f)
                    if selected_model in avancee_data:
                        model_data = avancee_data[selected_model]
                        global_metrics = model_data.get("Global", {})
                        per_query_metrics = model_data.get("PerQuery", {})
                        if global_metrics:
                            st.subheader(f"Overall Performance - {selected_model}")
                            cols = st.columns(min(4, len(global_metrics)))
                            for idx, (k, v) in enumerate(global_metrics.items()):
                                value = f"{v:.4f}" if isinstance(v, float) else str(v)
                                cols[idx % len(cols)].metric(k, value)
                        if str(selected_query) in per_query_metrics:
                            st.subheader(f"Extended Evaluation - Query {selected_query}")
                            query_metrics = per_query_metrics[str(selected_query)]
                            cols = st.columns(min(4, len(query_metrics)))
                            for idx, (k, v) in enumerate(query_metrics.items()):
                                value = f"{v:.4f}" if isinstance(v, float) else str(v)
                                cols[idx % len(cols)].metric(k, value)
                        else:
                            st.info("No metrics available for this query.")
                    else:
                        st.info("No metrics available for this model.")
                else:
                    st.info("Metrics file not found.")
                
                # Basic metrics section (Precision, Recall, F1)
                st.markdown("---")
                basic_metrics_path = Path("evaluation_results/basic_metrics/comparison_report.json")
                if basic_metrics_path.exists():
                    with open(basic_metrics_path, "r") as f:
                        basic_data = json.load(f)
                    
                    if selected_model in basic_data:
                        model_basic_data = basic_data[selected_model]
                        query_basic_metrics = model_basic_data.get("query_metrics", {}).get(str(selected_query), {})
                        
                        if query_basic_metrics:
                            st.subheader(f"Performance Summary - Query {selected_query}")
                            
                            # Display basic metrics in columns
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                precision = query_basic_metrics.get("precision", 0)
                                st.metric("Precision", f"{precision:.4f}")
                                
                                p_at_5 = query_basic_metrics.get("precision_at_5", 0)
                                st.metric("P@5", f"{p_at_5:.4f}")
                            
                            with col2:
                                recall = query_basic_metrics.get("recall", 0)
                                st.metric("Recall", f"{recall:.4f}")
                                
                                p_at_10 = query_basic_metrics.get("precision_at_10", 0)
                                st.metric("P@10", f"{p_at_10:.4f}")
                            
                            with col3:
                                f1_score = query_basic_metrics.get("f1_score", 0)
                                st.metric("F1 Score", f"{f1_score:.4f}")
                                
                                r_precision = query_basic_metrics.get("r_precision", 0)
                                st.metric("R-Precision", f"{r_precision:.4f}")
                        else:
                            st.info("No additional metrics for this query.")
                    else:
                        st.info("No additional metrics for this model.")
                else:
                    st.info("Additional metrics file not found.")
            else:
                st.info("Select a query and model to view metrics.")

    with tab3:
        if selected_query:
            display_all_models_comparison(selected_query)

    with tab4:
        if selected_model:
            display_pr_curves(selected_model)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><strong>Information Retrieval System</strong></p>
        <p>Advanced search and evaluation platform | Made with ❤️</p>
        <p style="font-size: 0.8rem; color: #90caf9;">Tip: Use the sidebar to explore different queries and models</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
