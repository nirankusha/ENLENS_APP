# -*- coding: utf-8 -*-
# =============================================================================
# Streamlit App: Document vs Corpus analysis (FlexiConc integration)
# - Document mode: run updated production pipeline on a single PDF (IG retained)
# - Corpus mode: scan directory of PDFs, process → store in FlexiConc SQLite,
#                browse corpus and drill into docs with overlays (IG retained)
# =============================================================================
import os, sys, json, time, sqlite3
from pathlib import Path

import streamlit as st
import pandas as pd

# Make sure we can import your updated modules (adjust paths if needed)
for p in ["/mnt/data", ".", "/content"]:
    if p not in sys.path:
        sys.path.append(p)

# --- Core imports (updated codebase) ---
from production_pipeline import run_complete_production_pipeline, streamlit_run
from ui_common import streamlit_select_sentence, render_sentence_overlay

# FlexiConc adapter (use if present; else fall back to simple stubs)
try:
    from flexiconc_adapter import (
        export_production_to_flexiconc, load_production_from_flexiconc, open_db
    )
    FLEXI_OK = True
except Exception as e:
    FLEXI_OK = False
    st.warning("FlexiConc adapter not found. Corpus mode will be limited.")
    def open_db(path): return sqlite3.connect(path)
    def export_production_to_flexiconc(*a, **k): pass
    def load_production_from_flexiconc(*a, **k): return {}

st.set_page_config(page_title="SDG Analyzer (FlexiConc)", layout="wide")

st.title("🎯 SDG Analyzer — Document & Corpus (FlexiConc)")

# ---------------- Sidebar ----------------
mode = st.sidebar.radio("Mode", ["Document analysis", "Corpus analysis"])

candidate_source = st.sidebar.selectbox(
    "Candidate source", ["span", "kpe"], index=0,
    help="span: SpanBERT-style spans; kpe: BERT-KPE phrases"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### IG & Overlays")
st.sidebar.write("Integrated Gradients is computed per sentence and rendered as a heat overlay. "
                 "Coreference spans are boxed. Output schema is identical across modes.")

# ---------------- Document Analysis ----------------
if mode == "Document analysis":
    st.subheader("Single document")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    max_sentences = st.number_input("Max sentences", 5, 500, 50, step=5)
    max_span_len  = st.number_input("Max span length (SpanBERT)", 1, 8, 4, step=1)
    top_k         = st.number_input("Top-K spans/phrases per sentence", 1, 20, 8, step=1)

    st.markdown("**Optional: Save to FlexiConc**")
    fc1, fc2, fc3 = st.columns([2,2,1])
    with fc1:
        db_path = st.text_input("SQLite DB path", "flexiconc.sqlite")
    with fc2:
        doc_id = st.text_input("Document ID", "uploaded.pdf")
    with fc3:
        save_to_db = st.checkbox("Save", value=False)

    if uploaded and st.button("Run analysis", type="primary", use_container_width=True):
        tmp_path = Path("tmp_upload.pdf")
        tmp_path.write_bytes(uploaded.read())

        prog = st.progress(0.0, text="Starting…")
        def _cb(msg, p): prog.progress(min(max(p,0.0),1.0), text=msg)

        out = run_complete_production_pipeline(
            str(tmp_path),
            candidate_source=candidate_source,
            max_sentences=int(max_sentences),
            max_span_len=int(max_span_len),
            top_k=int(top_k),
            progress_callback=_cb
        )

        st.success("Done.")
        # sentence picker + overlay
        sid = streamlit_select_sentence(st, out, source=("span" if candidate_source=="span" else "kp"))
        st.markdown(render_sentence_overlay(out, sid), unsafe_allow_html=True)

        if save_to_db and FLEXI_OK:
            export_production_to_flexiconc(db_path, doc_id, out, uri=uploaded.name, write_embeddings=False)
            st.info(f"Saved into {db_path} as doc_id='{doc_id}'.")

# ---------------- Corpus Analysis (FlexiConc) ----------------
else:
    st.subheader("Corpus from a directory → FlexiConc")
    colA, colB = st.columns([2,1])
    with colA:
        root_dir = st.text_input("Directory with PDFs", "/content/corpus")
    with colB:
        db_path = st.text_input("SQLite DB path", "flexiconc.sqlite")

    col1, col2, col3 = st.columns(3)
    with col1:
        max_sentences = st.number_input("Max sentences / doc", 5, 500, 50, step=5)
    with col2:
        max_span_len  = st.number_input("Max span len (SpanBERT)", 1, 8, 4, step=1)
    with col3:
        top_k         = st.number_input("Top-K spans/phrases", 1, 20, 8, step=1)

    st.caption("Corpus mode runs the same pipeline per file and persists results to FlexiConc. "
               "You can then browse the corpus and open any document with the same overlay.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📥 Index directory into FlexiConc", type="primary", use_container_width=True) and FLEXI_OK:
            root = Path(root_dir)
            pdfs = sorted([p for p in root.rglob("*.pdf")]) if root.exists() else []
            if not pdfs:
                st.error("No PDFs found in that directory.")
            else:
                pb = st.progress(0.0)
                for i, pdf in enumerate(pdfs, 1):
                    pb.progress(i/len(pdfs), text=f"{i}/{len(pdfs)}: {pdf.name}")
                    out = run_complete_production_pipeline(
                        str(pdf),
                        candidate_source=candidate_source,
                        max_sentences=int(max_sentences),
                        max_span_len=int(max_span_len),
                        top_k=int(top_k),
                        progress_callback=None
                    )
                    doc_id = pdf.stem
                    export_production_to_flexiconc(db_path, doc_id, out, uri=str(pdf), write_embeddings=False)
                st.success(f"Indexed {len(pdfs)} documents into {db_path}")

    # ---- Browse corpus from DB ----
    st.markdown("### 📚 Browse corpus")
    if FLEXI_OK:
        try:
            conn = open_db(db_path)
            docs = pd.read_sql_query("SELECT doc_id, uri, created_at FROM documents ORDER BY created_at DESC", conn)
            st.dataframe(docs, use_container_width=True, height=240)
            if not docs.empty:
                sel_doc = st.selectbox("Open document", docs["doc_id"].tolist())
                if sel_doc and st.button("Open", use_container_width=True):
                    loaded = load_production_from_flexiconc(db_path, sel_doc)
                    if loaded and loaded.get("sentence_analyses"):
                        sid = streamlit_select_sentence(st, loaded, source=("span" if candidate_source=="span" else "kp"), key="loaded_sel")
                        st.markdown(render_sentence_overlay(loaded, sid), unsafe_allow_html=True)
                    else:
                        st.info("No analyses found for that document.")
        except Exception as e:
            st.error(f"DB error: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass
    else:
        st.info("FlexiConc adapter unavailable. Add `flexiconc_adapter.py` to enable corpus browsing.")

"""
Created on Sat Aug 16 17:34:04 2025

@author: niran
"""

