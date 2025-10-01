# -*- coding: utf-8 -*-
# launch_flexiconc.py
# -----------------------------------------------------------------------------
# Launch Streamlit app with ngrok (production) or Colab iframe fallback.
# - Uses threading.Thread to open the ngrok tunnel without blocking.
# - Prints the public URL when ready.
# -----------------------------------------------------------------------------
# app_flexiconc.py
import os, sys, json
from pathlib import Path
import streamlit as st
import pandas as pd

# --- Stability env flags (safe on all platforms) ---
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Make sure we can import updated modules
for p in ["/mnt/data", ".", "/content"]:
    if p not in sys.path:
        sys.path.append(p)

from ui_common import streamlit_select_sentence, render_sentence_overlay
from production_pipeline import run_complete_production_pipeline

# FlexiConc adapter (optional)
FLEXI_OK = True
try:
    from flexiconc_adapter import export_production_to_flexiconc, load_production_from_flexiconc, open_db
except Exception as e:
    FLEXI_OK = False
    export_production_to_flexiconc = None
    load_production_from_flexiconc = None
    open_db = None

st.set_page_config(page_title="SDG Analyzer (FlexiConc)", layout="wide")
st.title("🎯 SDG Analyzer — Document & Corpus (FlexiConc)")

mode = st.sidebar.radio("Mode", ["Document analysis", "Corpus analysis"])
candidate_source = st.sidebar.selectbox("Candidate source", ["span", "kpe"], index=0)
st.sidebar.caption("IG heat and span boxes are identical across modes.")

# ---------------- Document mode ----------------
if mode == "Document analysis":
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    max_sentences = st.number_input("Max sentences", 5, 500, 50, step=5)
    max_span_len  = st.number_input("Max span length (SpanBERT)", 1, 8, 4, step=1)
    top_k         = st.number_input("Top-K spans/phrases", 1, 20, 8, step=1)

    st.markdown("**Optional: Save to FlexiConc**")
    colA, colB, colC = st.columns([2,2,1])
    with colA:
        db_path = st.text_input("SQLite DB path", "flexiconc.sqlite")
    with colB:
        doc_id = st.text_input("Document ID", "uploaded.pdf")
    with colC:
        save_to_db = st.checkbox("Save", value=False)

    if uploaded and st.button("Run analysis", type="primary", use_container_width=True):
        try:
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
            st.success("Analysis complete.")
            sid = streamlit_select_sentence(st, out, source=("span" if candidate_source=="span" else "kp"))
            st.markdown(render_sentence_overlay(out, sid), unsafe_allow_html=True)

            if save_to_db and FLEXI_OK and export_production_to_flexiconc:
                try:
                    export_production_to_flexiconc(db_path, doc_id, out, uri=uploaded.name)
                    st.info(f"Saved to {db_path} with doc_id='{doc_id}'.")
                except Exception as e:
                    st.warning(f"Save failed: {e}")

        except Exception as e:
            st.error(f"Run failed: {e}")

# ---------------- Corpus mode ----------------
else:
    col1, col2 = st.columns([2,1])
    with col1:
        root_dir = st.text_input("Directory with PDFs", "/content/corpus")
    with col2:
        db_path = st.text_input("SQLite DB path", "flexiconc.sqlite")

    c1, c2, c3 = st.columns(3)
    with c1:
        max_sentences = st.number_input("Max sentences / doc", 5, 500, 50, step=5, key="corpus_ms")
    with c2:
        max_span_len  = st.number_input("Max span len (SpanBERT)", 1, 8, 4, step=1, key="corpus_msl")
    with c3:
        top_k         = st.number_input("Top-K spans/phrases", 1, 20, 8, step=1, key="corpus_topk")

    st.caption("Process a folder of PDFs into FlexiConc. Then browse and open any doc with the same overlay.")

    colX, colY = st.columns(2)
    with colX:
        if st.button("📥 Index directory", type="primary", use_container_width=True):
            if not FLEXI_OK or not export_production_to_flexiconc:
                st.error("FlexiConc adapter not available in this environment.")
            else:
                try:
                    root = Path(root_dir)
                    pdfs = sorted([p for p in root.rglob("*.pdf")]) if root.exists() else []
                    if not pdfs:
                        st.error("No PDFs found.")
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
                            export_production_to_flexiconc(db_path, doc_id, out, uri=uploaded.name)
                        st.success(f"Indexed {len(pdfs)} documents into {db_path}")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    st.markdown("### 📚 Browse corpus")
    if FLEXI_OK and open_db:
        try:
            conn = open_db(db_path)
            try:
                df = pd.read_sql_query("SELECT doc_id, uri, created_at FROM documents ORDER BY created_at DESC", conn)
            finally:
                conn.close()
            st.dataframe(df, use_container_width=True, height=260)
            if not df.empty:
                sel = st.selectbox("Open document", df["doc_id"].tolist())
                if sel and st.button("Open", use_container_width=True):
                    try:
                        loaded = load_production_from_flexiconc(db_path, sel)
                        if loaded and loaded.get("sentence_analyses"):
                            sid = streamlit_select_sentence(st, loaded, source=("span" if candidate_source=="span" else "kp"), key=f"loaded_{sel}")
                            st.markdown(render_sentence_overlay(loaded, sid), unsafe_allow_html=True)
                        else:
                            st.info("No analyses found for that document.")
                    except Exception as e:
                        st.error(f"Load failed: {e}")
        except Exception as e:
            st.error(f"DB error: {e}")
    else:
        st.info("FlexiConc adapter unavailable. Add flexiconc_adapter.py to enable corpus browsing.")

"""
Created on Sat Aug 16 17:49:04 2025

@author: niran
"""

