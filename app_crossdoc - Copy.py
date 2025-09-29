# -*- coding: utf-8 -*-
# app_crossdoc.py
# Streamlit UI: document/corpus analysis with clickable spans/words,
# coref lists, concordance, and SciCo clustering/communities (no graph viz)

import re
import os
from pathlib import Path
import importlib
import streamlit as st
from typing import Dict, List, Any

# Local helpers / configs
from ui_config import (
    INGEST_CONFIG, SDG_CLASSIFIER_CONFIG, EXPLAIN_CONFIG,
    COREF_CONFIG, CORPUS_INDEX_CONFIG, SCICO_CONFIG, UI_CONFIG
)
from st_helpers import (
    make_sentence_selector, render_sentence_text_with_chips,
    render_coref_panel, render_clusters_panel, render_concordance_panel,
    toast_info, reset_terms_on_sentence_change,
    render_topk_chip_bar, handle_clicked_term, render_clickable_token_strip,
    commit_current_phrase, undo_last_token, clear_phrase_builder,
    remove_phrase, _ensure_query_builder
)
from ui_common import render_sentence_overlay

from bridge_runners import (
    run_ingestion_quick, rows_from_production, build_scico,
    build_concordance, pick_sentence_coref_groups
)
from utils_upload import save_uploaded_pdf
from helper_addons import build_ngram_trie
# (optional) co-occ, if you want to build it too
try:
    from helper_addons import build_cooc_graph  # only if present in your tree
    HAVE_COOCC = True
except Exception:
    HAVE_COOCC = False

from flexiconc_adapter import (
    open_db,
    export_production_to_flexiconc,
    upsert_doc_trie,
    upsert_doc_cooc,
    count_indices,
    list_index_sizes,
)

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
    ss.setdefault("current_doc_rows", None)  # cache for single-doc concordance
    ss.setdefault("source_root", "")         # NEW: for legacy relative paths

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

# ---------- Single-doc concordance (no DB) ----------
def _rows_from_production_local(production_output: Dict[str, Any], uri: str) -> List[Dict[str, Any]]:
    """Flatten the current analysis to rows consumable by concordance; adds path for display."""
    rows: List[Dict[str, Any]] = []
    if not production_output:
        return rows
    for s in production_output.get("sentence_analyses", []):
        rows.append({
            "path": uri,
            "start": s.get("doc_start"),
            "end": s.get("doc_end"),
            "text": s.get("sentence_text") or "",
            "sentence_id": s.get("sentence_id"),
        })
    return rows

def _filter_rows_by_terms(rows: List[Dict[str, Any]], terms: List[str], mode: str="AND") -> List[Dict[str, Any]]:
    """In-memory AND/OR substring match over 'text' (case-insensitive)."""
    if not rows or not terms:
        return []
    tnorm = [t.strip().lower() for t in terms if t and t.strip()]
    if not tnorm:
        return []
    out = []
    for r in rows:
        txt = (r.get("text") or "").lower()
        hits = [(t in txt) for t in tnorm]
        ok = all(hits) if mode == "AND" else any(hits)
        if ok:
            out.append(r)
    return out

def _bold_terms_html(s: str, terms: List[str]) -> str:
    """Bold term occurrences (case-insensitive, word-ish boundary)."""
    if not s or not terms:
        return s or ""
    out = s
    for t in sorted(set(t for t in terms if t), key=len, reverse=True):
        pat = re.compile(rf"(?i)\b{re.escape(t)}\b")
        out = pat.sub(lambda m: f"<b>{m.group(0)}</b>", out)
    return out


def _render_single_doc_concordance():
    prod = _get_production_output(st.session_state.get("results"))
    uri = (st.session_state.get("results") or {}).get("document_uri", "")
    rows = st.session_state.get("current_doc_rows") or _rows_from_production_local(prod, uri)

    st.markdown("---")
    st.subheader("Single-doc Concordance (fresh analysis)")
    if not rows:
        st.caption("Run **Analyze selected (fresh)** (Corpus) or **Run PDF analysis** (Document) first.")
        return

    qcol1, qcol2 = st.columns([3, 1])
    with qcol1:
        terms = st.session_state.get("query_terms", [])
        st.write("Selected terms:", ", ".join(terms) if terms else "—")
    
    with qcol2:
        cc_mode = st.radio("Mode", ["AND", "OR"], horizontal=True, key="single_doc_cc_mode")
    
    matches = _filter_rows_by_terms(rows, terms, mode=cc_mode)
   
    st.caption(f"Matches: {len(matches)} sentence(s)")
    if not matches:
        st.info("Click chips or tokens above to add terms.")
        return

    for r in matches[:300]:
        s = _bold_terms_html(r["text"], terms)
        st.markdown(
            f"<div style='margin:6px 0;'>"
            f"<code style='font-size:13px'>{s}</code>"
            f"<div style='color:#666;font-size:12px'>{r['path']} "
            f"[{r.get('start')}:{r.get('end')}] (sid={r.get('sentence_id')})</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    if len(matches) > 300:
        st.caption(f"…and {len(matches)-300} more not shown.")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Controls")

    # Mode
    doc_mode = st.radio(
        "Mode", options=["document", "corpus"], index=0, key="mode_radio",
        help="Document: analyze one PDF. Corpus: use FlexiConc index + SciCo across docs."
    )
    st.session_state["config_ui"]["mode"] = doc_mode

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
    
    if doc_mode == "document":
        
        up = st.file_uploader("Upload PDF", type=["pdf"])
        if up:
            pdf_path = save_uploaded_pdf(up)
            st.session_state["pdf_path"] = pdf_path
            if st.button("Run PDF analysis"):                
                cfg_ing = st.session_state["config_ingest"]
                cfg_ui  = st.session_state["config_ui"]
                res = run_ingestion_quick(
                    pdf_path,
                    max_sentences=cfg_ing["max_sentences"],
                    max_text_length=cfg_ing["max_text_length"],
                    candidate_source=cfg_ui.get("candidate_source", "span"),
                    coref_shortlist_mode="trie",   # if you want trie-only
                    )
                st.session_state["results"] = res.get("production_output")

    else:
        dbp = st.text_input("FlexiConc SQLite path", value=st.session_state["db_path"])
        st.session_state["db_path"] = dbp
        # NEW: legacy/relative path support
        st.session_state["source_root"] = st.text_input(
            "Source folder (optional, e.g. /content/en)",
            value=st.session_state.get("source_root", "")
        )

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
        cfg["min_abs_importance"] = st.slider("min_abs_importance (filter noise)", 0.0, 1.0, cfg.get("min_abs_importance", 0.10), 0.01)
        cfg["topk_tokens_chips"] = st.slider("topk_tokens_chips", 1, 24, cfg.get("topk_tokens_chips", 8))
        cfg["topk_spans_chips"] = st.slider("topk_spans_chips", 1, 24, cfg.get("topk_spans_chips", 6))
    # Knobs — Coreference
    with st.expander("Coreference (fastcoref)", expanded=False):
        cfg = st.session_state["config_coref"]
        cfg["engine"] = st.selectbox("engine", ["fastcoref", "lingmess"],
                             index=["fastcoref","lingmess"].index(cfg.get("engine","fastcoref")))
        cfg["device"] = st.text_input("device", value=cfg.get("device","cuda:0"))
        cfg["resolve_text"] = st.checkbox("resolve_text", value=cfg.get("resolve_text", True))

        cfg["scope"] = st.selectbox("scope", ["whole_document","windowed"],
                            index=0 if cfg["scope"]=="whole_document" else 1)
        if cfg["scope"] == "windowed":
            cfg["window_sentences"] = st.number_input("window_sentences", 3, 200, cfg["window_sentences"])
            cfg["window_stride"]    = st.number_input("window_stride", 1, 200, cfg["window_stride"])
        with st.expander("Coref shortlisting", expanded=False):
            cfg_coref = st.session_state["config_coref"]

            cfg_coref["coref_shortlist_mode"] = st.selectbox(
                             "Shortlist mode",
                             ["off","trie","cooc","both"],
                             index=["off","trie","cooc","both"].index(cfg_coref.get("coref_shortlist_mode","trie")),
                             help="off = no shortlist, trie = token-trie only, cooc = co-occ only, both = union"
                             )
            cfg_coref["coref_shortlist_topk"] = st.slider(
                 "Top-K candidates", 1, 200, cfg_coref.get("coref_shortlist_topk", 50)
                 )
            cfg_coref["coref_trie_tau"] = st.slider(
                "Trie τ (WJacc)", 0.00, 1.00, cfg_coref.get("coref_trie_tau", 0.18), 0.01
                )
            cfg_coref["coref_cooc_tau"] = st.slider(
                "Co-occ τ (cosine)", 0.00, 1.00, cfg_coref.get("coref_cooc_tau", 0.18), 0.01
                )
            cfg_coref["coref_use_pair_scorer"] = st.checkbox(
                "Use pair scorer", value=cfg_coref.get("coref_use_pair_scorer", False)
                )
            cfg_coref["coref_scorer_threshold"] = st.slider(
                "Pair scorer threshold", 0.0, 1.0, cfg_coref.get("coref_scorer_threshold", 0.25), 0.01
                )    
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
    if doc_mode == "document":
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
                cfg_coref = st.session_state["config_coref"]
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
        # -------------------- CORPUS MODE --------------------
        db_path = st.session_state["db_path"]
        source_root = (st.session_state.get("source_root") or "").strip()

        def _resolve_path(p: str) -> str:
            if not p:
                return p
            if os.path.isabs(p):
                return p
            # join relative to source_root if provided
            return os.path.normpath(os.path.join(source_root or "", p))

        available_docs = []
        if not Path(db_path).exists():
            st.warning("Set a valid SQLite path.")
        else:
            conn = None
            try:
                from flexiconc_adapter import open_db
                import pandas as pd
                from helper_addons import ensure_documents_table
                ensure_documents_table(db_path)
                conn = open_db(db_path)
                import pandas as pd
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
                st.caption(f"Tables detected: {tables['name'].tolist()}")
                try:
                    conn.execute("""
                                 CREATE TABLE IF NOT EXISTS indices (
                                     doc_id  TEXT NOT NULL,
                                     kind    TEXT NOT NULL,
                                     payload BLOB,
                                     PRIMARY KEY(doc_id, kind)
                                     )
                                 """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_indices_kind ON indices(kind)")
                    conn.commit()
                except Exception as e:
                    st.warning(f"Could not ensure `indices` table (DB may be read-only): {e}")

                # First try new schema: documents(uri, full_text)
                # Figure out which columns exist
                cols_info = pd.read_sql_query("PRAGMA table_info(documents)", conn)
                doc_cols = set(cols_info["name"].astype(str))

                # Build a safe SELECT
                sel_cols = ["doc_id", "uri"]
                if "created_at" in doc_cols:
                    sel_cols.append("created_at")
                else:
                    sel_cols.append("NULL AS created_at")

                # Prefer stored text_length; fall back to LENGTH(full_text) if present; else NULL
                if "text_length" in doc_cols:
                    sel_cols.append("text_length")
                elif "full_text" in doc_cols:
                    sel_cols.append("LENGTH(full_text) AS text_length")
                else:
                    sel_cols.append("NULL AS text_length")

                df = pd.read_sql_query(
                    f"SELECT {', '.join(sel_cols)} FROM documents ORDER BY COALESCE(created_at, '') DESC",
                    conn
                    )
   
                if not df.empty:
                    st.success(f"Corpus DB ready - {len(df)} documents found")
                    st.dataframe(df, use_container_width=True)
                    available_docs = df.to_dict("records")
                else:
                    # Fallback: legacy schema spans_file/files (path-only)
                    st.caption("Using legacy corpus schema (spans_file / files).")
                    # probe table name
                    tb = None
                    for cand in ("spans_file", "files", "documents"):
                        try:
                            pd.read_sql_query(f"SELECT 1 FROM {cand} LIMIT 1", conn)
                            tb = cand
                            break
                        except Exception:
                            pass
                    if not tb:
                        st.warning("Could not find legacy file table (spans_file/files).")
                    else:
                        # find likely columns
                        meta = pd.read_sql_query(f"PRAGMA table_info({tb})", conn)
                        cols = {r["name"] for _, r in meta.iterrows()}
                        id_col = "id" if "id" in cols else ("rowid" if "rowid" in cols else next(iter(cols)))
                        path_col = next((c for c in ("path","filepath","filename","name") if c in cols), None)
                        if not path_col:
                            st.warning(f"Table {tb} has no path-like column.")
                        else:
                            df2 = pd.read_sql_query(
                                f"SELECT {id_col} AS doc_id, {path_col} AS uri FROM {tb}",
                                conn
                            )
                            if df2.empty:
                                st.info("Legacy table is empty.")
                            else:
                                # resolve relative paths using source_root (if set)
                                df2["uri"] = df2["uri"].apply(_resolve_path)
                                df2["created_at"] = None
                                df2["text_length"] = None
                                st.success(f"Found {len(df2)} file entries in {tb}.")
                                df2["created_at"] = None
                                df2["text_length"] = None
                                show_cols = [c for c in df2.columns if not (c in ("created_at","text_length") and df2[c].isna().all())]
                                st.dataframe(df2[show_cols], use_container_width=True)
                                available_docs = df2.to_dict("records")
            except Exception as e:
                st.error(f"Error accessing corpus: {e}")
            finally:
                if conn is not None:
                    conn.close()

        st.session_state["available_docs"] = available_docs

        # Document selection dropdown
        selected_doc_id = st.session_state.get("selected_doc_id")
        if available_docs:
            doc_options = [
                f"{d['doc_id']} - {d.get('uri','no path')} ({d.get('text_length',0)} chars)"
                for d in available_docs
            ]
            selected_idx = st.selectbox(
                "Select document to analyze:",
                range(len(doc_options)),
                format_func=lambda i: doc_options[i],
                key="corpus_doc_selector",
            )
            if selected_idx is not None:
                selected_doc_id = available_docs[selected_idx]["doc_id"]
                st.session_state["selected_doc_id"] = selected_doc_id
                st.info(f"Selected: {selected_doc_id}")

        # Actions: load saved vs fresh analysis
        c1, c2 = st.columns([1, 1])
        with c1:
            load_btn = st.button("Load Document Analysis (from DB)")
        with c2:
            fresh_btn = st.button("Analyze selected (fresh)")

        # Load saved (FlexiConc) analysis if present
        if load_btn and selected_doc_id is not None:
            try:
                from flexiconc_adapter import load_production_from_flexiconc
                with st.spinner("Loading saved analysis from DB..."):
                    loaded_production = load_production_from_flexiconc(db_path, selected_doc_id)
                    if loaded_production and loaded_production.get("sentence_analyses"):
                        st.session_state["results"] = {"production_output": loaded_production}
                        st.session_state["current_doc_rows"] = None
                        st.success(f"Loaded analysis for {selected_doc_id} - {len(loaded_production['sentence_analyses'])} sentences")
                    else:
                        st.error("No analysis found for selected document")
            except Exception as e:
                st.error(f"Failed to load document: {e}")

        # FRESH in-memory analysis on the selected row's URI (no FlexiConc I/O)
        if fresh_btn and selected_doc_id is not None:
            if not available_docs:
                st.warning("No corpus rows loaded.")
            else:
                row = next((r for r in available_docs if r["doc_id"] == selected_doc_id), None)
                if not row:
                    st.warning("Could not resolve selected document row.")
                else:
                    uri = row.get("uri") or row.get("path") or row.get("filepath")
                    if not uri:
                        st.warning("Selected row has no uri/path.")
                    else:
                        with st.spinner(f"Analyzing file (fresh): {uri}"):
                            try:                                
                                cfg_ing  = st.session_state["config_ingest"]
                                cfg_ui   = st.session_state["config_ui"]
                                cfg_coref = st.session_state["config_coref"]
                                res = run_ingestion_quick(
                                    pdf_path=uri,
                                    max_sentences=cfg_ing["max_sentences"],
                                    max_text_length=cfg_ing["max_text_length"],
                                    candidate_source=cfg_ui.get("candidate_source", "span"),
                                    coref_shortlist_mode=cfg_coref["coref_shortlist_mode"],
                                    coref_shortlist_topk=cfg_coref["coref_shortlist_topk"],
                                    coref_trie_tau=cfg_coref["coref_trie_tau"],
                                    coref_cooc_tau=cfg_coref["coref_cooc_tau"],
                                    coref_use_pair_scorer=cfg_coref.get("coref_use_pair_scorer", False),
                                    coref_scorer_threshold=cfg_coref.get("coref_scorer_threshold", 0.25),
                                    )   
                                
                                # attach uri for display; cache rows for single-doc concordance
                                res = dict(res or {})
                                res["document_uri"] = uri
                                st.session_state["results"] = res
                                st.session_state["current_doc_rows"] = _rows_from_production_local(
                                    _get_production_output(res), uri
                                )
                                prod = _get_production_output(res)
                                n = len((prod or {}).get("sentence_analyses", []))
                                if res.get("ok") and n:
                                    st.success(f"✅ Fresh analysis complete. {n} sentences available.")
                                else:
                                    st.error(f"❌ Analysis failed or empty. {res.get('error','')}")
                            except Exception as e:
                                st.exception(e)

# -------------------- Step 2: Sentence & Keyword/Span Selection --------------------
with st.container(border=True):
    st.subheader("2) Sentence & Keyword/Span Selection")
    prod = _get_production_output(st.session_state.get("results"))

    sent_idx = st.session_state["selected_sentence_idx"]
    sent_obj = None
    if prod and len(prod.get("sentence_analyses", [])):
        sent_idx, sent_obj = make_sentence_selector(prod, sent_idx)

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
        _ensure_query_builder()

        # 2a) Inline overlay (visual, not clickable)
        prod = _get_production_output(st.session_state.get("results"))
        if prod:
            html_overlay = render_sentence_overlay(
                prod, st.session_state["selected_sentence_idx"],
                highlight_coref=True,
                box_spans=True,  # keep boxes for spans
            )
            st.markdown(html_overlay, unsafe_allow_html=True)

        # 2b) Compact chip bar (Top-K tokens + spans)
        clicked_term = render_topk_chip_bar(
            sent_obj,
            topk_tokens=st.session_state["config_explain"]["topk_tokens_chips"],
            topk_spans=st.session_state["config_explain"]["topk_spans_chips"],
            min_abs_importance=st.session_state["config_explain"]["min_abs_importance"],
            pos_threshold=st.session_state["config_explain"]["positive_thr"],
            neg_threshold=st.session_state["config_explain"]["negative_thr"],
            candidate_source=st.session_state["config_ui"]["candidate_source"],
        )
        if clicked_term:
            handle_clicked_term(clicked_term)
            toast_info(f"Added to phrase: {clicked_term}")

        if st.session_state["config_ui"]["clickable_tokens"]:
            clicked2 = render_clickable_token_strip(
                sent_obj,
                max_per_row=10,
                pos_threshold=st.session_state["config_explain"]["positive_thr"],
                neg_threshold=st.session_state["config_explain"]["negative_thr"],
            )
            if clicked2:
                handle_clicked_term(clicked2)
                toast_info(f"Added to phrase: {clicked2}")

        # 2c) Phrase builder panel
        qb = st.session_state["query_builder"]
        st.caption("🧱 Phrase builder (click chips to add)")
        st.code(qb["current_phrase"] or "—", language=None)
        c1, c2, c3, c4 = st.columns([1, 1, 2, 4])
        with c1:
            if st.button("Undo last"):
                undo_last_token()
        with c2:
            if st.button("Clear"):
                clear_phrase_builder()
        with c3:
            if st.button("Commit phrase ✅", type="primary"):
                commit_current_phrase()
        with c4:
            st.caption(f"{len(qb['active_tokens'])} tokens in current phrase")

        st.caption("✅ Committed phrases")
        if qb["phrases"]:
            cols = st.columns(min(6, max(2, len(qb["phrases"]))))
            for i, ph in enumerate(qb["phrases"]):
                with cols[i % len(cols)]:
                    st.write(f"`{ph}`")
                    if st.button("✕", key=f"rm_phrase_{i}"):
                        remove_phrase(i)
        else:
            st.write("--")

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
            try:
                coref_groups = pick_sentence_coref_groups(prod, st.session_state["selected_sentence_idx"])
                render_coref_panel(coref_groups, prod, st.session_state["config_ui"]["mode"])
            except Exception as e:
                st.exception(e)  # show error inline to debug
        else:
            st.info("No coreference chains for this sentence.")
            
with col4:
    with st.container(border=True):
        st.subheader("4) Concordance / Communities & Clusters")

        # A) Concordance (corpus mode only, DB-backed)
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

        # C) Single-doc in-memory concordance (works for both doc mode and fresh corpus analysis)
        _render_single_doc_concordance()

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Built on your existing modules: ENLENS_SpanBert_corefree_prod.py, scico_graph_pipeline.py, helper.py, flexiconc_adapter.py (if present).")

"""
Created on Tue Aug 26 2025
@author: niran
"""
