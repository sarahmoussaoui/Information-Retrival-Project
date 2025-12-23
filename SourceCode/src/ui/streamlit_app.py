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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .result-item {
        background-color: #ffffff;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
    }
    .rank-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">Information Retrieval System</h1>', unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar():
    """Render the sidebar with query and model selection."""
    st.sidebar.title("Configuration")
    
    # Load queries
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
    
    # Model selection
    st.sidebar.subheader("Select Model")
    available_models = get_available_models()
    selected_model = st.sidebar.selectbox(
        "Retrieval Model",
        options=available_models
    )
    
    # Top-K results
    st.sidebar.subheader("Display Options")
    top_k = st.sidebar.slider("Number of results to display", 5, 50, 20)
    
    return selected_query, selected_model, top_k


def display_ranked_results(query_id, model_name, documents, results, top_k=20):
    """Display ranked search results."""
    st.header(f"Ranked Results for Query {query_id} - {model_name}")
    
    query_results = results.get("queries", {}).get(str(query_id), [])
    
    if not query_results:
        st.warning("No results found for this query.")
        return
    
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
                st.markdown(f'<div class="result-item">', unsafe_allow_html=True)
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
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dcg = query_metrics.get("dcg@20", 0)
        st.metric("DCG@20", f"{dcg:.4f}")
    
    with col2:
        ndcg = query_metrics.get("ndcg@20", 0)
        st.metric("NDCG@20", f"{ndcg:.4f}")
    
    with col3:
        if "precision@10" in query_metrics:
            st.metric("Precision@10", f"{query_metrics['precision@10']:.4f}")
    
    with col4:
        if "recall@10" in query_metrics:
            st.metric("Recall@10", f"{query_metrics['recall@10']:.4f}")
    
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
    
    # Load comparison data
    comparison_path = Path("evaluation_results/evaluation_results_dcg_ndcg_gain/all_models_comparison.json")
    
    if not comparison_path.exists():
        st.warning("Comparison data not available.")
        return
    
    with open(comparison_path, 'r') as f:
        all_models_data = json.load(f)
    
    comparison_data = []
    for model_name, model_data in all_models_data.items():
        query_metrics = model_data.get("query_metrics", {}).get(str(query_id), {})
        if query_metrics:
            comparison_data.append({
                "Model": model_name,
                "DCG@20": query_metrics.get("dcg@20", 0),
                "NDCG@20": query_metrics.get("ndcg@20", 0)
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df = df.sort_values("NDCG@20", ascending=False)
        
        # Display table
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Bar chart comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(name='DCG@20', x=df['Model'], y=df['DCG@20']))
        fig.add_trace(go.Bar(name='NDCG@20', x=df['Model'], y=df['NDCG@20']))
        
        fig.update_layout(
            title=f"Model Performance Comparison - Query {query_id}",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)


def display_pr_curves(model_name):
    """Display Precision-Recall curves."""
    st.header("Precision-Recall Curves")
    
    curves_dir = Path("evaluation_results/evaluation_results_PRCurve_MAP_P@K_RR/Curves")
    
    if not curves_dir.exists():
        st.warning("PR curves directory not found.")
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
        pr_curve_path = curves_dir / f"PR_Curve_{file_suffix}.png"
        if pr_curve_path.exists():
            st.subheader("Precision-Recall Curve")
            st.image(str(pr_curve_path), use_container_width=True)
        else:
            st.warning(f"PR curve not found: {pr_curve_path.name}")
    
    with col2:
        # Interpolated PR Curve
        interp_pr_path = curves_dir / f"Interpolated_PR_Curve_{file_suffix}.png"
        if not interp_pr_path.exists():
            interp_pr_path = curves_dir / f"interpolated_PR_Curve_{file_suffix}.png"
        
        if interp_pr_path.exists():
            st.subheader("Interpolated PR Curve")
            st.image(str(interp_pr_path), use_container_width=True)
        else:
            st.warning(f"Interpolated PR curve not found")


def main():
    """Main application."""
    render_header()
    
    # Sidebar
    selected_query, selected_model, top_k = render_sidebar()
    
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
    
    with tab2:
        if selected_query and selected_model:
            metrics = load_metrics(selected_model)
            if metrics:
                display_metrics(selected_query, selected_model, metrics)
    
    with tab3:
        if selected_query:
            display_all_models_comparison(selected_query)
    
    with tab4:
        if selected_model:
            display_pr_curves(selected_model)


if __name__ == "__main__":
    main()
