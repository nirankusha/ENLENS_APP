# -*- coding: utf-8 -*-
# app_flexiconc_optimized.py - Resource-efficient version
import os, sys, gc
import streamlit as st
import pandas as pd
from pathlib import Path

# Optimize memory usage
st.set_page_config(
    page_title="SDG Analyzer", 
    layout="wide",
    initial_sidebar_state="collapsed"  # Save space
)

# Add caching decorator for heavy operations
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Load models once and cache them"""
    try:
        # Import only when needed
        from production_pipeline import run_complete_production_pipeline
        return True
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return False

@st.cache_data(show_spinner="Processing document...")
def process_document(_uploaded_file, candidate_source, max_sentences, max_span_len, top_k):
    """Cache document processing results"""
    from production_pipeline import run_complete_production_pipeline
    
    # Save uploaded file temporarily
    tmp_path = Path(f"tmp_{_uploaded_file.name}")
    tmp_path.write_bytes(_uploaded_file.read())
    
    try:
        # Process with progress tracking
        result = run_complete_production_pipeline(
            str(tmp_path),
            candidate_source=candidate_source,
            max_sentences=max_sentences,
            max_span_len=max_span_len,
            top_k=top_k,
            progress_callback=None  # Disable progress for caching
        )
        return result
    finally:
        # Cleanup
        if tmp_path.exists():
            tmp_path.unlink()
        gc.collect()  # Force garbage collection

def main():
    st.title("🎯 SDG Analyzer (Optimized)")
    
    # Load models first
    if not load_models():
        st.stop()
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        candidate_source = st.selectbox(
            "Analysis Type", 
            ["span", "kpe"], 
            index=0,
            help="span: SpanBERT spans; kpe: BERT-KPE phrases"
        )
        
        # Reduced limits for Colab
        max_sentences = st.slider("Max sentences", 5, 50, 20)  # Reduced from 500
        max_span_len = st.slider("Max span length", 1, 6, 3)   # Reduced from 8
        top_k = st.slider("Top-K per sentence", 1, 10, 5)     # Reduced from 20
        
        # Memory management
        if st.button("🧹 Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            gc.collect()
            st.success("Cache cleared!")
    
    # Main interface
    st.subheader("📄 Upload Document")
    
    uploaded = st.file_uploader(
        "Choose PDF file", 
        type=["pdf"],
        help="Max 10MB for Colab stability"
    )
    
    if uploaded:
        # Check file size
        file_size = len(uploaded.read())
        uploaded.seek(0)  # Reset file pointer
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            st.error("File too large! Please use files under 10MB for Colab.")
            st.stop()
        
        st.info(f"File size: {file_size / 1024 / 1024:.1f} MB")
        
        # Process button
        if st.button("🔍 Analyze Document", type="primary"):
            
            with st.spinner("Processing document..."):
                try:
                    # Use cached processing
                    result = process_document(
                        uploaded, 
                        candidate_source, 
                        max_sentences, 
                        max_span_len, 
                        top_k
                    )
                    
                    # Store in session state for UI
                    st.session_state['analysis_result'] = result
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.exception(e)  # Show full traceback for debugging
    
    # Display results if available
    if 'analysis_result' in st.session_state:
        display_results(st.session_state['analysis_result'], candidate_source)

def display_results(result, candidate_source):
    """Display analysis results with lightweight UI"""
    from ui_common import build_sentence_options, render_sentence_overlay
    
    st.subheader("📊 Analysis Results")
    
    # Summary stats
    n_sentences = len(result.get('sentence_analyses', []))
    n_chains = result.get('coreference_analysis', {}).get('num_chains', 0)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Sentences", n_sentences)
    col2.metric("Coref Chains", n_chains)
    col3.metric("Source", candidate_source.upper())
    
    # Sentence selector
    if n_sentences > 0:
        labels, indices = build_sentence_options(
            result, 
            source=("span" if candidate_source == "span" else "kp")
        )
        
        selected_label = st.selectbox(
            "🔍 Select sentence to examine:",
            labels,
            key="sentence_selector"
        )
        
        # Extract sentence ID
        if ":" in selected_label:
            sentence_id = int(selected_label.split(":", 1)[0])
            
            # Render sentence overlay
            html_content = render_sentence_overlay(result, sentence_id)
            st.markdown(html_content, unsafe_allow_html=True)
            
            # Show detailed analysis
            with st.expander("🔬 Detailed Analysis"):
                sentence_data = None
                for sa in result.get('sentence_analyses', []):
                    if sa.get('sentence_id') == sentence_id:
                        sentence_data = sa
                        break
                
                if sentence_data:
                    st.json(sentence_data)

if __name__ == "__main__":
    main()
"""
Created on Sat Aug 16 20:46:35 2025

@author: niran
"""

