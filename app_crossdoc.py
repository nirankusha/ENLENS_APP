# -*- coding: utf-8 -*-
# app_crossdoc.py
# Streamlit UI: document/corpus analysis with clickable spans/words,
# coref lists, concordance, and SciCo clustering/communities (no graph viz)

import os
from pathlib import Path
import importlib
import streamlit as st

# Local helpers / configs
from ui_config import (
    INGEST_CONFIG, SDG_CLASSIFIER_CONFIG, EXPLAIN_CONFIG,
    COREF_CONFIG, CORPUS_INDEX_CONFIG, SCICO_CONFIG, UI_CONFIG
)
from st_helpers import (
    make_sentence_selector, render_sentence_text_with_chips,
    render_coref_panel, render_clusters_panel, render_concordance_panel,
    toast_info, reset_terms_on_sentence_change
)
from bridge_runners import (
    run_ingestion_quick, rows_from_production, build_scico,
    build_concordance, pick_sentence_coref_groups
)
from utils_upload import save_uploaded_pdf

# -------------------- Page config --------------------
st.set_page_config(page_title="Cross-Doc SDG Explorer", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h2>📚 Cross-Doc SDG Explorer</h2>", unsafe_allow_html=True)
st.caption("Sentence picking → (click spans/words) → coref & concordance → SciCo communities/clusters (list view)")

# -------------------- State init --------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("config_ingest", dict(INGEST_CONFIG))
    ss.setdefault("config_sdg", dict(SDG_CLASSIFIER_CONFIG))
    ss.setdefault("config_explain", dict(EXPLAIN_CONFIG))
    ss.setdefault("config_coref", dict(COREF_CONFIG))
    ss.setdefault("config_corpus", dict(CORPUS_INDEX_CONFIG))
    ss.setdefault("config_scico", dict(SCICO_CONFIG))
    ss.setdefault("config_ui", dict(UI_CONFIG))

    ss.setdefault("results", None)        # may be production dict or {"production_output": ...}
    ss.setdefault("pdf_path", None)
    ss.setdefault("db_path", ss["config_corpus"]["sqlite_path"])

    ss.setdefault("selected_sentence_idx", 0)
    ss.setdefault("query_terms", [])
    ss.setdefault("last_action_ts", 0.0)

_init_state()

def _get_production_output(obj):
    """
    Accepts either:
      A) the production dict itself (new behavior),
      B) a wrapper {"production_output": {...}} (legacy).
    Returns the production dict or None.
    """
    if not obj:
        return None
    if isinstance(obj, dict) and "sentence_analyses" in obj:
        return obj
    if isinstance(obj, dict) and "production_output" in obj:
        return obj["production_output"]
    return None

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Controls")

    # Mode
    mode = st.radio(
        "Mode", options=["document", "corpus"], index=0, key="mode_radio",
        help="Document: analyze one PDF. Corpus: use FlexiConc index + SciCo across docs."
    )
    st.session_state["config_ui"]["mode"] = mode

    # Candidate source (spans vs KPE)
    cand_src = st.radio("Candidate source", options=["span", "kp"], index=0,
                        help="span = model spans; kp = keyphrases (BERT-KPE)")
    st.session_state["config_ui"]["candidate_source"] = cand_src

    # Tokens clickable?
    st.session_state["config_ui"]["clickable_tokens"] = st.checkbox(
        "Make all words clickable (in addition to spans/kps)",
        value=st.session_state["config_ui"]["clickable_tokens"]
    )

    # Layout preference for steps 3–4
    layout_side_by_side = st.checkbox("Place Coref & SciCo side-by-side", value=True)

    # Persistence hook (OFF by default)
    st.session_state["config_ui"]["persist_terms_across_sentences"] = st.checkbox(
        "Persist selected terms across sentence changes (advanced)",
        value=st.session_state["config_ui"]["persist_terms_across_sentences"]
    )

    # Upload / Corpus path
    if mode == "document":
        up = st.file_uploader("Upload PDF", type=["pdf"])
        if up:
            pdf_path = save_uploaded_pdf(up)
            st.session_state["pdf_path"] = pdf_path
            st.success(f"PDF uploaded to: {pdf_path}")
    else:
        dbp = st.text_input("FlexiConc SQLite path", value=st.session_state["db_path"])
        st.session_state["db_path"] = dbp

    # Knobs — Ingestion
    with st.expander("Files / Corpus upload", expanded=False):
        cfg = st.session_state["config_ingest"]
        cfg["max_text_length"] = st.number_input("max_text_length", 10_000, 2_000_000, cfg["max_text_length"], step=10_000)
        cfg["max_sentences"] = st.number_input("max_sentences", 5, 500, cfg["max_sentences"])
        cfg["min_sentence_len"] = st.number_input("min_sentence_len", 1, 200, cfg["min_sentence_len"])
        cfg["dedupe_near_duplicates"] = st.checkbox("dedupe_near_duplicates", value=cfg["dedupe_near_duplicates"])
        cfg["emit_offsets"] = st.checkbox("emit_offsets", value=cfg["emit_offsets"])

    # Knobs — SDG / Dual consensus
    with st.expander("SDG Classifier / Dual consensus", expanded=False):
        cfg = st.session_state["config_sdg"]
        cfg["bert_checkpoint"] = st.text_input("bert_checkpoint", value=cfg["bert_checkpoint"])
        cfg["sim_checkpoint"] = st.text_input("sim_checkpoint", value=cfg["sim_checkpoint"])
        cfg["agree_threshold"] = st.slider("agree_threshold", 0.0, 1.0, cfg["agree_threshold"], 0.01)
        cfg["disagree_threshold"] = st.slider("disagree_threshold", 0.0, 1.0, cfg["disagree_threshold"], 0.01)
        cfg["min_confidence"] = st.slider("min_confidence", 0.0, 1.0, cfg["min_confidence"], 0.01)

    # Knobs — Explainability/KPE thresholds (also used for pill colours)
    with st.expander("Explainability (IG & span masking / KPE)", expanded=False):
        cfg = st.session_state["config_explain"]
        cfg["ig_enabled"] = st.checkbox("ig_enabled", value=cfg["ig_enabled"])
        cfg["span_masking_enabled"] = st.checkbox("span_masking_enabled", value=cfg["span_masking_enabled"])
        cfg["max_span_len"] = st.slider("max_span_len (masking)", 1, 8, cfg["max_span_len"])
        cfg["top_k_spans"] = st.slider("top_k_spans (masking)", 1, 20, cfg["top_k_spans"])
        cfg["kpe_top_k"] = st.slider("kpe_top_k", 1, 50, cfg["kpe_top_k"])
        cfg["kpe_threshold"] = st.slider("kpe_threshold", 0.0, 1.0, cfg["kpe_threshold"], 0.01)
        cfg["positive_thr"] = st.slider("positive_thr (pill blue)", 0.0, 1.0, cfg.get("positive_thr", 0.15), 0.01)
        cfg["negative_thr"] = st.slider("negative_thr (pill red)", 0.0, 1.0, cfg.get("negative_thr", 0.20), 0.01)

    # Knobs — Coreference
    with st.expander("Coreference (fastcoref)", expanded=False):
        cfg = st.session_state["config_coref"]
        cfg["scope"] = st.selectbox(
            "scope", ["whole_document", "windowed"],
            index=0 if cfg["scope"] == "whole_document" else 1
        )
        if cfg["scope"] == "windowed":
            cfg["window_sentences"] = st.number_input("window_sentences", 3, 200, cfg["window_sentences"])
            cfg["window_stride"] = st.number_input("window_stride", 1, 200, cfg["window_stride"])

    # Knobs — SciCo shortlist/cluster/community/summaries
    with st.expander("SciCo (shortlist → cluster → communities → summaries)", expanded=False):
        cfg = st.session_state["config_scico"]
        st.subheader("Shortlist")
        cfg["use_shortlist"] = st.checkbox("use_shortlist", value=cfg["use_shortlist"])
        c1, c2 = st.columns(2)
        with c1:
            cfg["faiss_topk"] = st.number_input("faiss_topk", 5, 512, cfg["faiss_topk"])
            cfg["add_lsh"] = st.checkbox("add_lsh", value=cfg["add_lsh"])
            cfg["minhash_k"] = st.number_input("minhash_k", 2, 32, cfg["minhash_k"])
            cfg["use_coherence"] = st.checkbox("use_coherence (SGNLP)", value=cfg["use_coherence"])
        with c2:
            cfg["nprobe"] = st.number_input("nprobe", 1, 64, cfg["nprobe"])
            cfg["lsh_threshold"] = st.slider("lsh_threshold", 0.1, 0.95, cfg["lsh_threshold"], 0.01)
            cfg["cheap_len_ratio"] = st.slider("cheap_len_ratio", 0.0, 1.0, cfg["cheap_len_ratio"], 0.05)
            cfg["cheap_jaccard"] = st.slider("cheap_jaccard", 0.0, 1.0, cfg["cheap_jaccard"], 0.01)
            cfg["coherence_threshold"] = st.slider("coherence_threshold", 0.0, 1.0, cfg["coherence_threshold"], 0.01)

        st.subheader("Clustering & communities")
        c1, c2 = st.columns(2)
        with c1:
            cfg["clustering"] = st.selectbox(
                "clustering", ["auto", "kmeans", "torque", "both", "none"],
                index=["auto", "kmeans", "torque", "both", "none"].index(cfg["clustering"])
            )
            cfg["kmeans_k"] = st.number_input("kmeans_k", 2, 100, cfg["kmeans_k"])
            cfg["community_on"] = st.selectbox(
                "community_on", ["all", "corefer", "parent_child"],
                index=["all", "corefer", "parent_child"].index(cfg["community_on"])
            )
        with c2:
            cfg["community_method"] = st.selectbox(
                "community_method", ["greedy", "louvain", "leiden", "labelprop", "none"],
                index=["greedy", "louvain", "leiden", "labelprop", "none"].index(cfg["community_method"])
            )
            cfg["prob_threshold"] = st.slider("prob_threshold", 0.0, 1.0, cfg["prob_threshold"], 0.01)
            cfg["max_degree"] = st.number_input("max_degree", 5, 500, cfg["max_degree"])
            cfg["top_edges_per_node"] = st.number_input("top_edges_per_node", 5, 500, cfg["top_edges_per_node"])

        st.subheader("Summaries")
        cfg["summarize"] = st.checkbox("summarize", value=cfg["summarize"])
        c1, c2 = st.columns(2)
        with c1:
            cfg["summarize_on"] = st.selectbox(
                "summarize_on", ["community", "kmeans", "torque"],
                index=["community", "kmeans", "torque"].index(cfg["summarize_on"])
            )
            cfg["summary_methods"] = st.multiselect(
                "summary_methods", ["centroid", "xsum", "presumm"],
                default=cfg["summary_methods"]
            )
            cfg["xsum_sentences"] = st.number_input("xsum_sentences", 1, 5, cfg["xsum_sentences"])
        with c2:
            cfg["sdg_topk"] = st.number_input("sdg_topk", 1, 10, cfg["sdg_topk"])
            cfg["cross_encoder_model"] = st.text_input("cross_encoder_model", value=cfg["cross_encoder_model"])
            cfg["centroid_sim_threshold"] = st.slider("centroid_sim_threshold", 0.0, 1.0, cfg["centroid_sim_threshold"], 0.01)
            cfg["centroid_top_n"] = st.number_input("centroid_top_n", 1, 50, cfg["centroid_top_n"])
            cfg["centroid_store_vector"] = st.checkbox("centroid_store_vector", value=cfg["centroid_store_vector"])

    # UI toggles
    with st.expander("UI", expanded=False):
        cfg = st.session_state["config_ui"]
        cfg["auto_run_scico"] = st.checkbox("Auto-run SciCo on selection", value=cfg["auto_run_scico"])
        cfg["show_viz"] = st.checkbox("Show viz (disabled for now)", value=False, disabled=True)
        cfg["debug"] = st.checkbox("Debug", value=cfg["debug"])

# -------------------- Main: Steps 1 & 2 (full-width) --------------------
with st.container(border=True):
    st.subheader("1) Files / Corpus")
    if mode == "document":
        p = st.session_state.get("pdf_path")
        if p:
            st.caption(
                f"Path: {p} • exists={os.path.exists(p)} • "
                f"size={(os.path.getsize(p) if os.path.exists(p) else 0)} bytes"
            )
        else:
            st.info("Upload a PDF in the sidebar to enable analysis.")

        if st.button("Run PDF analysis", type="primary", disabled=not p):
            with st.spinner("Analyzing…"):
                res = run_ingestion_quick(
                    pdf_path=p,
                    max_sentences=st.session_state["config_ingest"]["max_sentences"],
                    max_text_length=st.session_state["config_ingest"]["max_text_length"],
                    # Add the candidate source selection
                    candidate_source=st.session_state["config_ui"]["candidate_source"],
                    ig_enabled=st.session_state["config_explain"]["ig_enabled"],
                    span_masking_enabled=st.session_state["config_explain"]["span_masking_enabled"],
                    max_span_len=st.session_state["config_explain"]["max_span_len"],
                    top_k_spans=st.session_state["config_explain"]["top_k_spans"],
                    kpe_top_k=st.session_state["config_explain"]["kpe_top_k"],
                    kpe_threshold=st.session_state["config_explain"]["kpe_threshold"],
                    coref_scope=st.session_state["config_coref"]["scope"],
                    coref_window_sentences=st.session_state["config_coref"].get("window_sentences"),
                    coref_window_stride=st.session_state["config_coref"].get("window_stride"),
                    # sdg consensus knobs:
                    agree_threshold=st.session_state["config_sdg"]["agree_threshold"],
                    disagree_threshold=st.session_state["config_sdg"]["disagree_threshold"],
                    min_confidence=st.session_state["config_sdg"]["min_confidence"],
                )

                st.session_state["results"] = res
                prod = (res or {}).get("production_output")
                n = len((prod or {}).get("sentence_analyses", []))
                if res.get("ok") and n:
                    st.success(f"✅ Analysis complete. {n} sentences available.")
                else:
                    st.error(f"❌ Analysis failed or empty. {res.get('error','')}")
            
    else:
        db_path = st.session_state["db_path"]
        if not Path(db_path).exists():
            st.warning("Set a valid SQLite path.")
        else:
            st.success("Corpus DB ready.")

with st.container(border=True):
    st.subheader("2) Sentence & Keyword/Span Selection")
    prod = _get_production_output(st.session_state.get("results"))
    sent_idx, sent_obj = make_sentence_selector(prod, st.session_state["selected_sentence_idx"])

    # default: reset terms when sentence changes, unless persistence is enabled
    reset_terms_on_sentence_change(
        sent_idx,
        key_selected_idx="selected_sentence_idx",
        key_terms="query_terms",
        persist_flag_key="persist_terms_across_sentences",
    )
    st.session_state["selected_sentence_idx"] = sent_idx

    if not sent_obj:
        st.info("Run analysis (document) or switch to corpus mode.")
    else:
        st.write("**DEBUG: Sentence object keys:**", list(sent_obj.keys()))
        if "span_analysis" in sent_obj:
            st.write("**DEBUG: span_analysis:**", sent_obj["span_analysis"])
        if "token_analysis" in sent_obj:
            st.write("**DEBUG: token_analysis:**", sent_obj["token_analysis"])
        
        clicked_term = render_sentence_text_with_chips(
            sent_obj,
            candidate_source=st.session_state["config_ui"]["candidate_source"],
            clickable_tokens=st.session_state["config_ui"]["clickable_tokens"],
            pos_threshold=st.session_state["config_explain"]["positive_thr"],
            neg_threshold=st.session_state["config_explain"]["negative_thr"],
            kpe_top_k=st.session_state["config_explain"]["kpe_top_k"],
            kpe_threshold=st.session_state["config_explain"]["kpe_threshold"],
            debug=True
        )
        if clicked_term and clicked_term not in st.session_state["query_terms"]:
            st.session_state["query_terms"].append(clicked_term)
            toast_info(f"Added term: {clicked_term}")

        st.caption("Selected terms")
        c1, c2 = st.columns([3, 1])
        with c1:
            st.code(", ".join(st.session_state["query_terms"]) or "—")
        with c2:
            if st.button("Clear terms"):
                st.session_state["query_terms"] = []

# -------------------- Steps 3 & 4 (side-by-side or stacked) --------------------
if layout_side_by_side:
    col3, col4 = st.columns([1, 1], vertical_alignment="top")
else:
    col3, col4 = st.container(), st.container()

with col3:
    with st.container(border=True):
        st.subheader("3) Coreference")
        prod = _get_production_output(st.session_state.get("results"))
        if prod and len(prod.get("sentence_analyses", [])):
            coref_groups = pick_sentence_coref_groups(prod, st.session_state["selected_sentence_idx"])
            render_coref_panel(coref_groups, prod, st.session_state["config_ui"]["mode"])
        else:
            st.info("No coreference chains for this sentence.")

with col4:
    with st.container(border=True):
        st.subheader("4) Concordance / Communities & Clusters")

        # A) Concordance (corpus mode only)
        if st.session_state["config_ui"]["mode"] == "corpus":
            db_path = st.session_state["db_path"]
            if Path(db_path).exists() and st.session_state["query_terms"]:
                with st.spinner("Querying concordance…"):
                    conc = build_concordance(db_path, st.session_state["query_terms"], and_mode=True)
                render_concordance_panel(conc)
            else:
                st.info("Add terms and make sure DB path is valid.")

        # B) SciCo (document & corpus; list only, no viz)
        run_now = st.button("Run SciCo (using selected terms)")
        if (st.session_state["config_ui"]["auto_run_scico"] and st.session_state["query_terms"]) or run_now:
            prod = _get_production_output(st.session_state.get("results"))
            rows = rows_from_production(prod)
            if not rows:
                st.warning("No rows to run SciCo on.")
            else:
                with st.spinner("Running SciCo…"):
                    G, meta = build_scico(
                        rows=rows,
                        selected_terms=st.session_state["query_terms"],
                        scico_cfg=st.session_state["config_scico"],
                    )
                render_clusters_panel(
                    G, meta,
                    sentence_idx=st.session_state["selected_sentence_idx"],
                    summarize_opts={
                        "show_representative": True,
                        "show_xsum": "xsum" in st.session_state["config_scico"]["summary_methods"],
                        "show_presumm": "presumm" in st.session_state["config_scico"]["summary_methods"],
                    },
                )
        else:
            st.caption("SciCo: add terms and click the button (or enable auto-run).")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Built on your existing modules: ENLENS_SpanBert_corefree_prod.py, scico_graph_pipeline.py, helper.py, flexiconc_adapter.py (if present).")

"""
Created on Tue Aug 26 2025
@author: niran
"""
