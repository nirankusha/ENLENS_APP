# -*- coding: utf-8 -*-
# app_crossdoc.py
# Streamlit UI: document/corpus analysis with clickable spans/words,
# coref lists, concordance, and SciCo clustering/communities (no graph viz)

import re
import os
import time
import html
from pathlib import Path
import streamlit as st
from typing import Dict, List, Any, Optional, Iterable

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in some envs
    torch = None

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
    run_ingestion_quick, build_scico,
    build_concordance, pick_sentence_coref_groups
)
from utils_upload import save_uploaded_pdf
from helper import encode_mpnet, encode_sdg_hidden, encode_scico
# (writer bits used for corpus/global indices)
from global_coref_helper import global_coref_query, _make_grams, _idf_weighted_jaccard

from cooc_utils import resolve_cooc_backend, cooc_backend_ready

from flexiconc_adapter import (
    open_db, export_production_to_flexiconc,
    upsert_doc_trie, upsert_doc_cooc,
    count_indices, list_index_sizes,
    build_faiss_indices,
)
from helper_addons import build_ngram_trie
try:
    from helper_addons import build_cooc_graph
    _HAVE_COOCC_LOCAL = True
except Exception:
    _HAVE_COOCC_LOCAL = False

DEFAULT_EMBEDDING_MODELS = {
    "mpnet": encode_mpnet,
    "sdg-bert": encode_sdg_hidden,
    "scico": encode_scico,
}
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
    ss.setdefault("source_root", "")         # for legacy relative paths
    ss.setdefault("last_uri", None)          # remember last analyzed file path for save/upsert
    ss.setdefault("corpus_concordance", None)    
    
    ss.setdefault("auto_run_global_coref", False)
    ss.setdefault("global_coref_hits", [])
    ss.setdefault("global_coref_and_mode", "OR")
    ss.setdefault("global_coref_tau_trie", 0.18)
    ss.setdefault("global_coref_topk", 50)
    ss.setdefault("filter_local_coref", False)
    ss.setdefault("local_chain_grams", None)
    ss.setdefault("scico_panel_data", None)
    
    ss.setdefault("sidebar_pdf_dir", "")
    ss.setdefault("sidebar_new_db_path", "")
    ss.setdefault("sidebar_enable_cooc", False)
    ss.setdefault("sidebar_coref_mode", "trie")
_init_state()

def _available_coref_devices() -> List[str]:
    """Return the set of supported coref devices in preferred order."""
    options = ["cpu"]
    if torch is not None:
        try:
            if torch.cuda.is_available():
                count = torch.cuda.device_count() or 1
                options.extend([f"cuda:{i}" for i in range(count)])
        except Exception:
            pass
    return options


def _default_coref_device() -> str:
    opts = _available_coref_devices()
    for opt in opts:
        if opt.startswith("cuda"):
            return opt
    return opts[0]


def _normalize_coref_device(device_val: Optional[str]) -> str:
    """Normalize UI-provided device strings to supported values."""
    requested = (device_val or "").strip()
    opts = _available_coref_devices()
    default = _default_coref_device()
    if not requested:
        return default
    lowered = requested.lower()
    if lowered in {"auto", "default", ""}:
        return default
    if lowered == "cuda":
        # normalize bare cuda to the first GPU option
        return default if default.startswith("cuda") else "cpu"
    for opt in opts:
        if lowered == opt.lower():
            return opt
    if lowered.startswith("cuda:") and any(opt.startswith("cuda") for opt in opts):
        return default
    return default


def _ensure_coref_device(cfg_coref: Dict[str, Any]) -> str:
    """Ensure the config dict carries a supported device string."""
    norm = _normalize_coref_device(cfg_coref.get("device"))
    cfg_coref["device"] = norm
    return norm

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

# ===== Resolved sentence helpers & SciCo tree prep ======

def _current_doc_label() -> Optional[str]:
    res = st.session_state.get("results")
    if isinstance(res, dict):
        for key in ("document_id", "doc_id", "document_uri", "uri"):
            val = res.get(key)
            if val:
                return str(val)
        prod = res.get("production_output") if "production_output" in res else res
        if isinstance(prod, dict):
            for key in ("document_id", "doc_id", "document_uri", "uri"):
                val = prod.get(key)
                if val:
                    return str(val)
    return None

def _collect_sentence_ranges(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranges = []
    for s in sentences or []:
        sid = s.get("sentence_id")
        ranges.append({
            "sentence_id": int(sid) if sid is not None else None,
            "start": s.get("doc_start"),
            "end": s.get("doc_end"),
        })
    return ranges

def _assign_sentence_id(mention: Dict[str, Any], ranges: List[Dict[str, Any]]) -> Optional[int]:
    sid = mention.get("sentence_id")
    if sid is not None:
        try:
            return int(sid)
        except Exception:
            return None
    s0, s1 = mention.get("start_char"), mention.get("end_char")
    for r in ranges:
        a, b = r.get("start"), r.get("end")
        if None in (a, b, s0, s1):
            continue
        if int(a) <= int(s0) and int(s1) <= int(b):
            return r.get("sentence_id")
    return None

def _resolve_sentence_texts(production_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    sentences = production_output.get("sentence_analyses") or []
    chains = (production_output.get("coreference_analysis") or {}).get("chains") or []
    ranges = _collect_sentence_ranges(sentences)

    replacements: Dict[int, List[Dict[str, Any]]] = {}
    for ch in chains:
        rep = (ch.get("representative") or "").strip()
        for mention in ch.get("mentions") or []:
            sid = _assign_sentence_id(mention, ranges)
            if sid is None:
                continue
            if not mention.get("is_pronoun", False):
                continue
            s0, s1 = mention.get("start_char"), mention.get("end_char")
            if None in (s0, s1):
                continue
            replacements.setdefault(int(sid), []).append({
                "start_char": int(s0),
                "end_char": int(s1),
                "replacement": rep,
            })

    resolved: List[Dict[str, Any]] = []
    for sent in sentences:
        text = sent.get("sentence_text") or ""
        sid = sent.get("sentence_id")
        try:
            sid_int = int(sid) if sid is not None else None
        except Exception:
            sid_int = None
        doc_start = sent.get("doc_start")
        doc_end = sent.get("doc_end")
        items = []
        for repl in replacements.get(sid_int, []):
            s0 = repl.get("start_char")
            s1 = repl.get("end_char")
            if None in (s0, s1, doc_start, doc_end):
                continue
            local_start = int(s0) - int(doc_start)
            local_end = int(s1) - int(doc_start)
            if local_start < 0 or local_end > len(text) or local_start >= local_end:
                continue
            items.append((local_start, local_end, repl.get("replacement") or ""))
        items.sort(key=lambda it: it[0], reverse=True)
        resolved_text = text
        for s0, s1, rep in items:
            rep = rep or ""
            if s0 < 0 or s1 > len(resolved_text) or s0 >= s1:
                continue
            resolved_text = resolved_text[:s0] + rep + resolved_text[s1:]
        resolved.append({
            "sid": sid_int,
            "resolved_text": resolved_text,
            "original_text": text,
            "doc_start": doc_start,
            "doc_end": doc_end,
        })
    return resolved

def _rows_from_resolved_sentences(resolved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in resolved or []:
        rows.append({
            "sid": item.get("sid"),
            "text": item.get("resolved_text") or "",
            "path": "",
            "start": item.get("doc_start"),
            "end": item.get("doc_end"),
        })
    return rows

def _safe_label(meta_val, idx: int) -> Optional[int]:
    if meta_val is None:
        return None
    try:
        if isinstance(meta_val, dict):
            return None if meta_val.get(idx) is None else int(meta_val.get(idx))
        if isinstance(meta_val, (list, tuple)):
            return None if idx >= len(meta_val) else int(meta_val[idx])
        return int(meta_val[idx])
    except Exception:
        return None

def _build_chain_trees(
    G,
    meta: Dict[str, Any],
    rows: List[Dict[str, Any]],
    production_output: Dict[str, Any],
    resolved_sentences: List[Dict[str, Any]],
    *,
    include_chains: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    chains = (production_output.get("coreference_analysis") or {}).get("chains") or []
    sid_to_idx: Dict[int, int] = {}
    for idx, row in enumerate(rows):
        sid = row.get("sid")
        if sid is None:
            continue
        try:
            sid_to_idx[int(sid)] = idx
        except Exception:
            continue

    resolved_by_sid = {item.get("sid"): item for item in resolved_sentences}

    chain_nodes: Dict[int, Dict[str, Any]] = {}
    for ch in chains:
        try:
            cid = int(ch.get("chain_id"))
        except Exception:
            continue
        node_indices: set[int] = set()
        for mention in ch.get("mentions") or []:
            sid = mention.get("sentence_id")
            if sid is None:
                continue
            try:
                idx = sid_to_idx.get(int(sid))
            except Exception:
                idx = None
            if idx is not None:
                node_indices.add(idx)
        if not node_indices:
            continue
        chain_nodes[cid] = {
            "indices": sorted(node_indices),
            "representative": ch.get("representative") or "",
        }

    if include_chains is not None:
        include_set = {int(c) for c in include_chains}
        chain_nodes = {cid: data for cid, data in chain_nodes.items() if cid in include_set}

    if not chain_nodes:
        return []

    parent_children: Dict[int, set[int]] = {}
    incoming: Dict[int, set[int]] = {}
    corefers: Dict[int, set[int]] = {}
    for u, v, data in G.edges(data=True):
        label = data.get("label")
        if label == "corefer":
            corefers.setdefault(int(u), set()).add(int(v))
        elif label in {"parent", "child"}:
            parent_children.setdefault(int(u), set()).add(int(v))
            incoming.setdefault(int(v), set()).add(int(u))

    trees: List[Dict[str, Any]] = []
    for cid, info in sorted(chain_nodes.items(), key=lambda kv: kv[0]):
        nodes_set = set(info["indices"])
        if not nodes_set:
            continue

        children_map: Dict[int, List[int]] = {}
        for parent, kids in parent_children.items():
            if parent not in nodes_set:
                continue
            children = sorted(k for k in kids if k in nodes_set)
            if children:
                children_map[parent] = children

        incoming_map: Dict[int, set[int]] = {}
        for child in nodes_set:
            parents = incoming.get(child, set())
            incoming_map[child] = {p for p in parents if p in nodes_set}

        roots = sorted(n for n in nodes_set if not incoming_map.get(n))
        if not roots:
            roots = sorted(nodes_set)

        coref_map: Dict[int, List[int]] = {}
        for node in nodes_set:
            neigh = sorted(n for n in corefers.get(node, set()) if n in nodes_set and n != node)
            if neigh:
                coref_map[node] = neigh

        nodes_payload: Dict[int, Dict[str, Any]] = {}
        for node in nodes_set:
            row = rows[node]
            sid = row.get("sid")
            resolved_info = resolved_by_sid.get(sid)
            nodes_payload[node] = {
                "sentence_idx": node,
                "sentence_id": sid,
                "resolved_text": (resolved_info or {}).get("resolved_text") or row.get("text") or "",
                "original_text": (resolved_info or {}).get("original_text") or row.get("text") or "",
                "doc_start": row.get("start"),
                "doc_end": row.get("end"),
                "community": _safe_label(meta.get("communities"), node),
                "kmeans": _safe_label(meta.get("kmeans"), node),
                "torque": _safe_label(meta.get("torque"), node),
            }

        trees.append({
            "chain_id": cid,
            "representative": info.get("representative") or "",
            "roots": roots,
            "children_map": children_map,
            "coref_map": coref_map,
            "nodes": nodes_payload,
            "all_nodes": sorted(nodes_set),
        })

    return trees

def _prepare_scico_panel(
    production_output: Optional[Dict[str, Any]],
    terms: List[str],
    *,
    source: str,
    doc_label: Optional[str] = None,
    include_chains: Optional[Iterable[int]] = None,
    is_current_doc: bool = False,
    precomputed_embeddings: Any = None,
    embedding_provider: Optional[Callable[[List[str]], Any]] = None,
) -> Dict[str, Any]:
    panel: Dict[str, Any] = {
        "source": source,
        "doc_label": doc_label,
        "is_current_doc": is_current_doc,
    }
    terms_norm = [t.strip() for t in terms or [] if t and t.strip()]
    panel["terms"] = terms_norm
    panel["focus_chains"] = list(include_chains) if include_chains else None

    if not production_output:
        panel["error"] = "No production output available for SciCo."
        return panel

    resolved_sentences = _resolve_sentence_texts(production_output)
    rows_resolved = _rows_from_resolved_sentences(resolved_sentences)
    panel["resolved_sentences"] = resolved_sentences
    panel["rows"] = rows_resolved

    if not terms_norm:
        panel["warning"] = "Commit phrases to trigger SciCo runs."
        return panel
    if not rows_resolved:
        panel["error"] = "No sentences available for SciCo."
        return panel

    try:
        G, meta = build_scico(
            rows=rows_resolved,
            selected_terms=terms_norm,
            scico_cfg=st.session_state["config_scico"],
        )
    except Exception as exc:
        panel["error"] = f"SciCo failed: {exc}"
        return panel

    panel["graph"] = G
    panel["meta"] = meta
    trees = _build_chain_trees(
        G,
        meta,
        rows_resolved,
        production_output,
        resolved_sentences,
        include_chains=include_chains,
    )
    panel["chains"] = trees
    if not trees:
        panel.setdefault("warning", "SciCo produced no chain-aligned sentences for the selected phrases.")
    return panel

def _render_sentence_entry(idx: int, chain: Dict[str, Any], panel: Dict[str, Any], level: int, relation: str):
    node = chain["nodes"].get(idx)
    if not node:
        return

    indent_px = max(0, int(level) * 18)
    relation_map = {
        "parent": "Parent",
        "child": "Child",
        "corefer": "Corefer",
    }
    relation_label = relation_map.get(relation, "Sentence")

    sid = node.get("sentence_id")
    sid_display = "—" if sid is None else sid
    cluster_bits = []
    if node.get("community") is not None:
        cluster_bits.append(f"comm {node['community']}")
    if node.get("kmeans") is not None:
        cluster_bits.append(f"kmeans {node['kmeans']}")
    if node.get("torque") is not None:
        cluster_bits.append(f"torque {node['torque']}")
    cluster_text = " · ".join(cluster_bits)

    resolved_preview = html.escape(_preview_text(node.get("resolved_text"), 180))
    original_preview = html.escape(_preview_text(node.get("original_text"), 180))
    if resolved_preview != original_preview:
        body_html = (
            f"<div class='mono'>{resolved_preview}</div>"
            f"<div style='font-size:11px;color:#7a7a7a'>orig: {original_preview}</div>"
        )
    else:
        body_html = f"<div class='mono'>{resolved_preview}</div>"

    header_html = f"{relation_label} · Sentence {sid_display}"
    if cluster_text:
        header_html += f" · {cluster_text}"

    key_suffix = f"{panel.get('doc_label','doc')}_{chain['chain_id']}_{idx}_{relation}_{level}"
    focus_target = sid if sid is not None else node.get("sentence_idx")

    with st.container():
        text_col, action_col = st.columns([12, 1])
        with text_col:
            st.markdown(
                f"""
                <div style='margin-left:{indent_px}px;border-left:1px solid #e0e0e0;padding-left:10px;margin-bottom:6px;'>
                  <div style='font-size:12px;color:#555;'>{header_html}</div>
                  <div style='font-size:13px;'>{body_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with action_col:
            if st.button("Focus", key=f"focus_{key_suffix}"):
                if focus_target is not None:
                    st.session_state["selected_sentence_idx"] = int(focus_target)
                st.session_state["last_action_ts"] = time.time()

def _render_chain_node(idx: int, chain: Dict[str, Any], panel: Dict[str, Any], visited: set[int], level: int, relation: str):
    if idx in visited:
        return
    visited.add(idx)
    _render_sentence_entry(idx, chain, panel, level, relation)

    for neighbor in chain.get("coref_map", {}).get(idx, []):
        if neighbor in visited:
            continue
        _render_chain_node(neighbor, chain, panel, visited, level, "corefer")

    for child in chain.get("children_map", {}).get(idx, []):
        if child in visited:
            continue
        _render_chain_node(child, chain, panel, visited, level + 1, "child")

def _render_chain_hierarchy(panel_data: Optional[Dict[str, Any]]):
    st.markdown("#### SciCo · Chain hierarchy")
    if not panel_data:
        st.caption("Chain-centric trees appear after committing phrases or sending a global hit to the panel.")
        return

    if panel_data.get("doc_label") or panel_data.get("terms"):
        bits = []
        if panel_data.get("doc_label"):
            bits.append(f"Doc: `{panel_data['doc_label']}`")
        if panel_data.get("terms"):
            bits.append("Terms: " + ", ".join(f"`{t}`" for t in panel_data["terms"]))
        if panel_data.get("source"):
            bits.append(f"Source: {panel_data['source']}")
        st.caption(" · ".join(bits))

    if panel_data.get("error"):
        st.error(panel_data["error"])
        return

    if panel_data.get("warning"):
        st.info(panel_data["warning"])

    chains = panel_data.get("chains") or []
    if not chains:
        return

    _ensure_highlight_css()

    for chain in chains:
        header = f"Chain {chain['chain_id']} · {len(chain['all_nodes'])} sentence(s)"
        rep = chain.get("representative")
        if rep:
            header += f" · rep=“{_preview_text(rep, 60)}”"
        with st.expander(header, expanded=False):
            visited: set[int] = set()
            for root in chain.get("roots") or []:
                _render_chain_node(root, chain, panel_data, visited, level=0, relation="parent")
            leftovers = [idx for idx in chain.get("all_nodes") if idx not in visited]
            if leftovers:
                st.caption("Unattached sentences")
                for idx in leftovers:
                    _render_chain_node(idx, chain, panel_data, visited, level=0, relation="corefer")

def _handle_phrase_commit():
    prod = _get_production_output(st.session_state.get("results"))
    panel = _prepare_scico_panel(
        prod,
        st.session_state.get("query_terms", []),
        source="doc_phrase_commit",
        doc_label=_current_doc_label(),
        include_chains=None,
        is_current_doc=True,
    )
    st.session_state["scico_panel_data"] = panel
    if panel.get("error"):
        st.toast(panel["error"], icon="❌")
    elif panel.get("warning"):
        st.toast(panel["warning"], icon="ℹ️")
    else:
        st.toast("SciCo hierarchy updated for committed phrases.", icon="✅")

# === Local coref helpers (doc-level) =========================================
def _build_local_chain_grams(production_output):
    """
    Build and cache grams for each chain in the current analysis.
    Stored in st.session_state['local_chain_grams'] as {chain_id: Counter}.
    """
    if not production_output:
        st.session_state["local_chain_grams"] = {}
        return

    chains = (production_output.get("coreference_analysis") or {}).get("chains") or []
    chain_grams = {}
    for ch in chains:
        chain_id = ch.get("chain_id")
        rep = (ch.get("representative") or "").strip()
        if not rep and ch.get("mentions"):
            # fallback to first mention text if present
            rep = (ch["mentions"][0].get("text") or "")
        chain_grams[int(chain_id)] = _make_grams(rep, char_ns=(4,), token_ns=(2,3))
    st.session_state["local_chain_grams"] = chain_grams


def _filter_local_coref_by_phrases(production_output, phrases, tau_doc=0.15):
    """
    Return a ranked list of chain_ids that match committed phrases by IDF-weighted Jaccard.
    For doc-level filtering we can use uniform idf (1.0) to keep it simple.
    """
    if not production_output or not phrases:
        return []

    # Ensure cache exists
    if st.session_state.get("local_chain_grams") is None:
        _build_local_chain_grams(production_output)

    chain_grams = st.session_state.get("local_chain_grams") or {}
    if not chain_grams:
        return []

    # score each chain; if multiple phrases, OR=max, AND=min
    mode = st.session_state.get("global_coref_and_mode", "OR")
    idf_doc = {}  # uniform weights
    scores = []

    # Pre-make grams for phrases
    G_list = [_make_grams(p, char_ns=(4,), token_ns=(2,3)) for p in phrases if p.strip()]

    for cid, cg in chain_grams.items():
        if mode == "AND":
            # all phrases must match; aggregate via min
            per = [_idf_weighted_jaccard(G, cg, idf_doc) for G in G_list]
            s = min(per) if per else 0.0
        else:
            # OR: any phrase can match; aggregate via max
            per = [_idf_weighted_jaccard(G, cg, idf_doc) for G in G_list]
            s = max(per) if per else 0.0
        if s >= tau_doc:
            scores.append({"chain_id": cid, "score": float(s)})
    scores.sort(key=lambda r: r["score"], reverse=True)
    return scores


# === Corpus/global coref (Trie) ==============================================
def _run_global_coref_from_phrases(db_path, phrases, and_mode="OR", tau_trie=0.18, topk=50):
    """
    Runs global trie query for committed phrases, aggregates (AND/OR), returns ranked hits.
    """
    if not Path(db_path).exists() or not phrases:
        return []

    # Run per-phrase, then aggregate
    from flexiconc_adapter import open_db
    cx = open_db(db_path)

    def _one(phrase):
        return global_coref_query(phrase, cx, use_trie=True, use_cooc=False, topk=topk, tau_trie=tau_trie) or []

    buckets = [ _one(p) for p in phrases if p.strip() ]
    cx.close()

    if not buckets:
        return []

    # Index hits as (doc_id, chain_id) -> list of scores
    from collections import defaultdict
    coll = defaultdict(list)

    if and_mode == "AND":
        # Only chains present in all buckets
        key_sets = [ {(h["doc_id"], int(h["chain_id"])) for h in b} for b in buckets ]
        must = set.intersection(*key_sets) if key_sets else set()
        for b in buckets:
            for h in b:
                key = (h["doc_id"], int(h["chain_id"]))
                if key in must:
                    coll[key].append(float(h.get("score_trie", h.get("score", 0.0))))
        # aggregate via min (strong AND semantics)
        agg = [
            {"doc_id": d, "chain_id": c, "score": min(vals), "score_trie": min(vals), "why": "trie"}
            for (d,c), vals in coll.items()
        ]
    else:
        # OR: union; aggregate via max
        for b in buckets:
            for h in b:
                key = (h["doc_id"], int(h["chain_id"]))
                coll[key].append(float(h.get("score_trie", h.get("score", 0.0))))
        agg = [
            {"doc_id": d, "chain_id": c, "score": max(vals), "score_trie": max(vals), "why": "trie"}
            for (d,c), vals in coll.items()
        ]

    agg.sort(key=lambda r: r["score"], reverse=True)
    return agg[:topk]


# === Snippet fetcher ==========================================================
def _fetch_chain_snippets(db_path, doc_id, chain_id, limit=3):
    """
    Pull short snippets for (doc_id, chain_id) using chains.mentions_json and sentences.
    """
    import json, sqlite3
    snips = []
    cx = sqlite3.connect(db_path)
    cur = cx.cursor()
    try:
        mrow = cur.execute(
            "SELECT mentions_json FROM chains WHERE doc_id=? AND chain_id=?",
            (doc_id, int(chain_id))
        ).fetchone()
        if not mrow or not mrow[0]:
            cx.close()
            return []

        mentions = json.loads(mrow[0]) or []
        # Map sentence ranges for quick lookup
        sent_rows = cur.execute(
            "SELECT start, end, text FROM sentences WHERE doc_id=?",
            (doc_id,)
        ).fetchall()
        # For each mention (start,end), pick containing sentence if any
        for m in mentions:
            s0 = m.get("start"); s1 = m.get("end")
            txt = m.get("text") or ""
            found = None
            for (a,b,t) in sent_rows:
                if s0 is not None and s1 is not None and a is not None and b is not None:
                    if (s0 >= a) and (s1 <= b):
                        found = t; break
            snips.append(found or txt)
            if len(snips) >= limit:
                break
        return [s for s in snips if s]
    except Exception:
        return []
    finally:
        cx.close()

# --- small text preview & CSS for highlights ---
def _preview_text(s: str, n=160) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[:n].rstrip() + "…"

def _ensure_highlight_css():
    st.markdown("""
    <style>
      .hl { padding: 0 3px; border-radius: 3px; background: rgba(255, 235, 59, 0.45); }
      .hl2 { padding: 0 2px; border-bottom: 2px solid #ff9800; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    </style>
    """, unsafe_allow_html=True)

# --- make foldable list/dict blocks like PRAGMA output ---
def tree(label: str, payload, *, open=False):
    if isinstance(payload, dict):
        with st.expander(f"{label} · keys={len(payload)}", expanded=open):
            st.write(payload)
    elif isinstance(payload, (list, tuple)):
        with st.expander(f"{label} · items={len(payload)}", expanded=open):
            st.write(payload)
    else:
        st.write(f"**{label}:**", payload)

# --- sentence highlighter (for #5) ---
def _highlight_sentence(text: str, spans):
    """
    text: full sentence string
    spans: list of (start,end) offsets RELATIVE to this sentence
    """
    if not text:
        return ""
    spans = [(int(a or 0), int(b or 0)) for (a,b) in spans if a is not None and b is not None and a < b]
    spans.sort()
    out, i = [], 0
    for a, b in spans:
        a = max(0, min(a, len(text)))
        b = max(0, min(b, len(text)))
        if a > i: out.append(text[i:a])
        if b > a: out.append(f'<span class="hl">{text[a:b]}</span>')
        i = max(i, b)
    if i < len(text): out.append(text[i:])
    return "".join(out)

# --- pull sentences + relative-mention spans for a (doc_id, chain_id) ---
def _fetch_chain_sentences_with_mentions(db_path, doc_id, chain_id, limit=6):
    import sqlite3, json
    cx = sqlite3.connect(db_path)
    cur = cx.cursor()
    try:
        row = cur.execute("SELECT mentions_json FROM chains WHERE doc_id=? AND chain_id=?",
                          (doc_id, int(chain_id))).fetchone()
        if not row or not row[0]:
            return []
        mentions = json.loads(row[0]) or []

        sents = cur.execute("SELECT sentence_id, start, end, text FROM sentences WHERE doc_id=? ORDER BY sentence_id",
                            (doc_id,)).fetchall()
        out = []
        for sid, a, b, t in sents:
            local = []
            for m in mentions:
                s0, s1 = m.get("start"), m.get("end")
                if s0 is None or s1 is None or a is None or b is None: 
                    continue
                if a <= s0 and s1 <= b:
                    local.append((s0 - a, s1 - a))
            if local:
                out.append({"sid": sid, "text": t, "spans": local})
                if len(out) >= limit:
                    break
        return out
    finally:
        cx.close()

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

# ---------- MULTI-DOC CONCORDANCE HELPERS (GLOBAL TRIE) ----------
def _kwic_from_sentence(text: str, spans, width=50):
    """Return list of (L, KW, R) KWIC triples for spans in this sentence."""
    out = []
    if not text:
        return out
    for a, b in spans:
        a = max(0, min(a, len(text)))
        b = max(0, min(b, len(text)))
        L = text[max(0, a - width):a]
        KW = text[a:b]
        R = text[b:min(len(text), b + width)]
        out.append((L, KW, R))
    return out

def _fetch_chain_sentences_with_mentions(db_path, doc_id, chain_id, limit=None):
    """
    Returns: [{sid, text, spans=[(rel_a, rel_b), ...]}], ordered by sid.
    """
    import sqlite3, json
    cx = sqlite3.connect(db_path)
    cur = cx.cursor()
    try:
        row = cur.execute("SELECT mentions_json FROM chains WHERE doc_id=? AND chain_id=?",
                          (doc_id, int(chain_id))).fetchone()
        if not row or not row[0]:
            return []
        mentions = json.loads(row[0]) or []

        sents = cur.execute(
            "SELECT sentence_id, start, end, text FROM sentences WHERE doc_id=? ORDER BY sentence_id",
            (doc_id,)
        ).fetchall()

        out = []
        for sid, a, b, t in sents:
            local = []
            for m in mentions:
                s0, s1 = m.get("start"), m.get("end")
                if None in (s0, s1, a, b):
                    continue
                if a <= s0 and s1 <= b:
                    local.append((s0 - a, s1 - a))
            if local:
                out.append({"sid": sid, "text": t, "spans": local})
        if limit:
            out = out[:limit]
        return out
    finally:
        cx.close()

def _ensure_highlight_css():
    st.markdown("""
    <style>
      .kwic { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
      .kwL { opacity: .75; }
      .kwK { padding: 0 3px; background: rgba(255,235,59,.45); border-radius: 3px; }
      .kwR { opacity: .75; }
      .tight { margin-top: .2rem; margin-bottom: .2rem; }
    </style>
    """, unsafe_allow_html=True)

def _render_kwic_lines(sentence_obj, width=50):
    """Render KWIC lines for one sentence object from _fetch_chain_sentences_with_mentions."""
    s = sentence_obj
    for L, KW, R in _kwic_from_sentence(s["text"], s["spans"], width=width):
        st.markdown(
            f'<div class="kwic tight"><span class="kwL">{L}</span>'
            f'<span class="kwK">{KW}</span><span class="kwR">{R}</span></div>',
            unsafe_allow_html=True
        )

def render_global_trie_multi_concordance(*, hits, db_path, kwic_width=50, max_sentences_per_doc=80):
    """
    hits: iterable of dicts with at least {'doc_id', 'chain_id', 'score'}
    db_path: sqlite file path
    """
    _ensure_highlight_css()

    # group hits by doc
    by_doc = {}
    for h in (hits or []):
        d = h.get("doc_id"); c = h.get("chain_id")
        if d is None or c is None: 
            continue
        by_doc.setdefault(d, []).append(h)

    st.markdown("### Multi-document concordance (global trie)")
    st.caption(f"{len(by_doc)} document(s) with matches")

    for doc_id, doc_hits in by_doc.items():
        # a header with total # chains and cumulated score info
        chains = sorted(doc_hits, key=lambda x: -float(x.get("score", 0.0)))
        score_info = ", ".join([f"#{h['chain_id']}={h.get('score', 0):.3f}" for h in chains[:4]])
        with st.expander(f"{doc_id} · chains={len(chains)} · {score_info}{' …' if len(chains) > 4 else ''}", expanded=False):

            # nav: open in existing views (keep your current UX)
            cols = st.columns(2)
            with cols[0]:
                st.link_button(f"Open in Concordance: {doc_id}", f"#", help="Use your existing concordance hook")
            with cols[1]:
                st.link_button(f"Open doc analysis: {doc_id}", f"#", help="Use your existing doc-analysis hook")

            # per-chain foldables with KWIC
            total_rendered = 0
            for h in chains:
                cid = h["chain_id"]; sc = float(h.get("score", 0.0))
                with st.expander(f"Chain {cid} · score={sc:.3f}", expanded=False):
                    rows = _fetch_chain_sentences_with_mentions(db_path, doc_id, cid)
                    st.caption(f"{len(rows)} sentence(s) with mentions")
                    for r in rows:
                        st.markdown(f"**Sentence {r['sid']}**")
                        _render_kwic_lines(r, width=kwic_width)
                        total_rendered += 1
                        if total_rendered >= max_sentences_per_doc:
                            st.info("Truncated view (too many lines). Increase `max_sentences_per_doc` to see more.")
                            break
                if total_rendered >= max_sentences_per_doc:
                    break



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
    
    st.session_state["auto_run_global_coref"] = st.checkbox(
            "Auto-run Global Coref (Trie) on commit", value=st.session_state["auto_run_global_coref"]
        )
    st.session_state["filter_local_coref"] = st.checkbox(
            "Filter local coref by committed phrases", value=st.session_state["filter_local_coref"]
        )
    ingest_submit = False
    if doc_mode == "document":
        up = st.file_uploader("Upload PDF", type=["pdf"])
        if up:
            pdf_path = save_uploaded_pdf(up)
            st.session_state["pdf_path"] = pdf_path
            if st.button("Run PDF analysis"):
                cfg_ing = st.session_state["config_ingest"]
                cfg_ui  = st.session_state["config_ui"]
                cfg_coref = st.session_state["config_coref"]
                cooc_mode_val, cooc_tok, cooc_warn = resolve_cooc_backend(cfg_coref)
                res = run_ingestion_quick(
                    pdf_path,
                    max_sentences=cfg_ing["max_sentences"],
                    max_text_length=cfg_ing["max_text_length"],
                    candidate_source=cfg_ui.get("candidate_source", "span"),
                    coref_backend=cfg_coref.get("engine", "fastcoref"),
                    coref_device = _ensure_coref_device(cfg_coref),
                    coref_scope=cfg_coref.get("scope"),
                    coref_window_sentences=cfg_coref.get("window_sentences"),
                    coref_window_stride=cfg_coref.get("window_stride"),
                    resolve_text=cfg_coref.get("resolve_text"),
                    coref_shortlist_mode=cfg_coref.get("coref_shortlist_mode", "trie"),
                    coref_shortlist_topk=cfg_coref.get("coref_shortlist_topk"),
                    coref_trie_tau=cfg_coref.get("coref_trie_tau"),
                    coref_cooc_tau=cfg_coref.get("coref_cooc_tau"),
                    coref_use_pair_scorer=cfg_coref.get("coref_use_pair_scorer"),
                    coref_scorer_threshold=cfg_coref.get("coref_scorer_threshold"),
                    coref_pair_scorer=cfg_coref.get("coref_pair_scorer"),
                    cooc_mode=cooc_mode_val,
                    cooc_hf_tokenizer=cooc_tok,
                )
                st.session_state["results"] = res.get("production_output")
                st.session_state["last_uri"] = pdf_path
    else:
        dbp_raw = st.text_input("FlexiConc SQLite path", value=st.session_state["db_path"])
        dbp = os.path.expanduser(dbp_raw.strip()) if dbp_raw else dbp_raw
        if dbp:
            st.session_state["db_path"] = dbp

        ss = st.session_state
        ss.setdefault("source_root", ss.get("source_root", ""))
        ss.setdefault("sidebar_pdf_dir_input", ss.get("source_root", ""))
        ss.setdefault("sidebar_new_db_dest_input", "")
        ss.setdefault("sidebar_enable_cooc", False)
        ss.setdefault("sidebar_coref_mode", "trie")
        
        with st.form("flexiconc_ingest_sidebar"):
            st.text_input(
                "PDF folder to ingest",
                value=st.session_state.get("sidebar_pdf_dir", ""),
                key="sidebar_pdf_dir",
            )
            st.text_input(
                "New FlexiConc destination/name (optional)",
                value=st.session_state.get("sidebar_new_db_path", ""),
                key="sidebar_new_db_path",
                help="Provide a file name or path for a new SQLite DB. Leave blank to append to the current DB.",
            )
            c_ingest_a, c_ingest_b = st.columns(2)
            with c_ingest_a:
                st.checkbox(
                    "Also build co-occurrence",
                    key="sidebar_enable_cooc",
                    help="Requires SciPy CSR support (helper_addons).",
                    disabled=not _HAVE_COOCC_LOCAL,
                )
            with c_ingest_b:
                st.selectbox(
                    "Coref shortlist mode",
                    options=["trie", "both", "none"],
                    key="sidebar_coref_mode",
                    help="Passed to run_ingestion_quick for trie/co-occ shortlist generation.",
                )
            ingest_submit = st.form_submit_button("Ingest folder into FlexiConc")
        # legacy/relative path support
        ss["source_root"] = st.text_input(
            "Source folder (optional, e.g. /content/en)",
            value=ss.get("source_root", "")
        )
        with st.form("corpus_ingest_form", clear_on_submit=False):
            pdf_dir_val = st.text_input(
                "PDF folder to ingest",
                key="sidebar_pdf_dir_input",
                help="Folder containing PDFs to add/update in the FlexiConc database.",
            )
            new_db_val = st.text_input(
                "New FlexiConc destination/name (optional)",
                key="sidebar_new_db_dest_input",
                help="Provide a new SQLite path to create/populate a fresh FlexiConc database.",
            )
            enable_cooc_sidebar = st.checkbox(
                "Also build co-occurrence (requires SciPy)",
                key="sidebar_enable_cooc",
                help="If available, export co-occurrence graphs for each document.",
                disabled=not _HAVE_COOCC_LOCAL,
            )
            coref_mode_sidebar = st.selectbox(
                "Coref shortlist mode",
                options=["trie", "both", "none"],
                index=["trie", "both", "none"].index(ss.get("sidebar_coref_mode", "trie")),
                key="sidebar_coref_mode",
                help="Passed to run_ingestion_quick for trie/co-occurrence shortlist generation.",
            )
            ingest_submit = st.form_submit_button("Ingest folder into FlexiConc")

        cfg_corpus = ss["config_corpus"]
        c1, c2 = st.columns([3, 1])
        with c1:
            cfg_corpus["vector_backend"] = st.text_input(
                "Embedding backend (mpnet or ST model)",
                value=cfg_corpus.get("vector_backend", "mpnet"),
                help="Provide an alias like 'mpnet' or a sentence-transformers model id for concordance vector search.",
            )
        with c2:
            cfg_corpus["use_faiss"] = st.checkbox(
                "Use FAISS",
                value=cfg_corpus.get("use_faiss", True),
                help="Enable FAISS acceleration when a pre-built embedding index is available.",
            )
        ingest_status_placeholder = st.empty()

        if ingest_submit:
            target_pdf_dir = Path(st.session_state.get("sidebar_pdf_dir_input", "")).expanduser()
            dest_override = st.session_state.get("sidebar_new_db_dest_input", "").strip()
            target_db = Path(dest_override or st.session_state.get("db_path", "")).expanduser()

            if not target_pdf_dir or not str(target_pdf_dir):
                ingest_status_placeholder.error("Please provide a PDF folder to ingest.")
            elif not target_pdf_dir.exists() or not target_pdf_dir.is_dir():
                ingest_status_placeholder.error(f"PDF folder not found: {target_pdf_dir}")
            else:
                try:
                    target_db.parent.mkdir(parents=True, exist_ok=True)
                    cx0 = open_db(str(target_db))
                    cx0.close()
                except Exception as e:
                    ingest_status_placeholder.error(f"Unable to open or create FlexiConc DB: {e}")
                else:
                    if dest_override:
                        st.session_state["db_path"] = str(target_db)
                        st.session_state["sidebar_new_db_dest_input"] = ""

                    cfg_coref = st.session_state["config_coref"]
                    cooc_mode_val, cooc_tok, cooc_warn = resolve_cooc_backend(cfg_coref)
                    if cooc_warn and st.session_state.get("sidebar_enable_cooc"):
                        st.warning(cooc_warn)

                    pdfs = sorted(target_pdf_dir.rglob("*.pdf"))
                    if not pdfs:
                        ingest_status_placeholder.warning("No PDFs found in that folder.")
                    else:
                        progress = st.progress(0.0, text="Starting ingestion…")
                        done = errs = skipped = 0
                        total = len(pdfs)
                        for i, path_item in enumerate(pdfs, start=1):
                            this_uri = str(path_item)
                            this_doc_id = Path(path_item).stem
                            progress.progress(i / total, text=f"[{i}/{total}] {this_doc_id}")
                            try:
                                res = run_ingestion_quick(
                                    pdf_path=this_uri,
                                    max_sentences=st.session_state["config_ingest"]["max_sentences"],
                                    max_text_length=st.session_state["config_ingest"]["max_text_length"],
                                    candidate_source=st.session_state["config_ui"].get("candidate_source", "span"),
                                    coref_backend=cfg_coref.get("engine", "fastcoref"),
                                    coref_device=_ensure_coref_device(cfg_coref),
                                    resolve_text=cfg_coref.get("resolve_text", True),
                                    coref_scope=cfg_coref.get("scope"),
                                    coref_window_sentences=cfg_coref.get("window_sentences"),
                                    coref_window_stride=cfg_coref.get("window_stride"),
                                    coref_shortlist_mode=st.session_state.get("sidebar_coref_mode", "trie"),
                                    coref_shortlist_topk=cfg_coref["coref_shortlist_topk"],
                                    coref_trie_tau=cfg_coref["coref_trie_tau"],
                                    coref_cooc_tau=cfg_coref["coref_cooc_tau"],
                                    coref_use_pair_scorer=cfg_coref.get("coref_use_pair_scorer", False),
                                    coref_scorer_threshold=cfg_coref.get("coref_scorer_threshold", 0.25),
                                    coref_pair_scorer=cfg_coref.get("coref_pair_scorer"),
                                    cooc_mode=cooc_mode_val,
                                    cooc_hf_tokenizer=cooc_tok,
                                )
                                prod = res.get("production_output") or res

                                export_production_to_flexiconc(
                                    str(target_db),
                                    this_doc_id,
                                    prod,
                                    uri=this_uri,
                                    embedding_models=DEFAULT_EMBEDDING_MODELS,
                                )

                                chains = (prod.get("coreference_analysis") or {}).get("chains") or []
                                if chains:
                                    trie_root, trie_idf, chain_grams = build_ngram_trie(chains, char_n=4, token_ns=(2, 3))

                                    cx2 = open_db(str(target_db))
                                    try:
                                        upsert_doc_trie(cx2, this_doc_id, trie_root, trie_idf, chain_grams)

                                        if st.session_state.get("sidebar_enable_cooc") and _HAVE_COOCC_LOCAL:
                                            full_text = prod.get("full_text") or ""
                                            if full_text:
                                                vocab, rows_c, norms = build_cooc_graph(
                                                    full_text,
                                                    window=5,
                                                    min_count=2,
                                                    topk_neighbors=10,
                                                    mode=cooc_mode_val,
                                                    hf_tokenizer=cooc_tok,
                                                )
                                                upsert_doc_cooc(cx2, this_doc_id, vocab, rows_c, norms)
                                        cx2.commit()
                                    finally:
                                        cx2.close()
                                    done += 1
                                else:
                                    skipped += 1
                                    ingest_status_placeholder.info(f"No chains in analysis for {this_doc_id}; trie not written.")
                            except Exception as e:
                                errs += 1
                                ingest_status_placeholder.exception(e)

                        progress.progress(1.0, text="Finishing up…")
                        try:
                            cx3 = open_db(str(target_db))
                            try:
                                n_now = count_indices(cx3, "trie")
                                summary = build_faiss_indices(cx3)
                                ingest_status_placeholder.success(
                                    f"Ingestion complete. Trie rows now: {n_now}. OK={done}, skipped={skipped}, errors={errs}."
                                )
                                if summary:
                                    ingest_status_placeholder.caption(
                                        ", ".join(
                                            f"{k}: {v.get('count', 0)} vecs ({v.get('model')})"
                                            for k, v in summary.items()
                                        )
                                    )
                                try:
                                    ingest_status_placeholder.caption(
                                        f"Sample trie payload sizes: {list_index_sizes(cx3, 'trie', limit=3)}"
                                    )
                                except Exception:
                                    pass
                            finally:
                                cx3.close()
                        except Exception as e:
                            ingest_status_placeholder.warning(
                                f"Documents ingested but FAISS index build failed: {e}"
                            )

                        progress.empty()

                        st.session_state["sidebar_pdf_dir_input"] = str(target_pdf_dir)
        if ingest_submit:
            pdf_dir_val = (st.session_state.get("sidebar_pdf_dir") or "").strip()
            new_db_val = (st.session_state.get("sidebar_new_db_path") or "").strip()
            enable_cooc_sidebar = bool(st.session_state.get("sidebar_enable_cooc", False) and _HAVE_COOCC_LOCAL)
            coref_mode_sidebar = (st.session_state.get("sidebar_coref_mode") or "trie")

            if not pdf_dir_val:
                st.warning("Provide a PDF folder to ingest.")
            else:
                pdf_dir_path = Path(pdf_dir_val).expanduser()
                if not pdf_dir_path.exists() or not pdf_dir_path.is_dir():
                    st.error(f"PDF folder not found: {pdf_dir_path}")
                else:
                    current_db_path = Path(st.session_state.get("db_path") or cfg_corpus.get("sqlite_path", "")).expanduser()
                    target_db_path = current_db_path
                    if new_db_val:
                        candidate = Path(new_db_val).expanduser()
                        if not candidate.is_absolute():
                            candidate = (current_db_path.parent / candidate).expanduser()
                        target_db_path = candidate

                    target_parent = target_db_path.parent
                    try:
                        target_parent.mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        st.error(f"Unable to create parent folder {target_parent}: {exc}")
                        target_db_path = None

                    if target_db_path is not None:
                        try:
                            cx_test = open_db(str(target_db_path))
                            cx_test.close()
                        except Exception as exc:
                            st.exception(exc)
                            target_db_path = None

                    if target_db_path is not None:
                        target_db_path_str = str(target_db_path)
                        if new_db_val:
                            st.session_state["db_path"] = target_db_path_str
                            cfg_corpus["sqlite_path"] = target_db_path_str
                            st.session_state["sidebar_new_db_path"] = target_db_path_str

                        pdfs = sorted(pdf_dir_path.rglob("*.pdf"))
                        if not pdfs:
                            st.warning("No PDFs found in that folder.")
                        else:
                            cfg_ing = st.session_state["config_ingest"]
                            cfg_ui = st.session_state["config_ui"]
                            cfg_coref = st.session_state["config_coref"]
                            cooc_mode_val, cooc_tok, cooc_warn = resolve_cooc_backend(cfg_coref)

                            log_container = st.container()
                            progress = st.progress(0.0, text="Starting ingestion…")
                            done = errs = 0
                            total = len(pdfs)

                            if enable_cooc_sidebar and cooc_warn:
                                log_container.warning(f"{cooc_warn} Co-occurrence exports will use the spaCy backend.")

                            for i, path_item in enumerate(pdfs, start=1):
                                this_uri = str(path_item)
                                this_doc_id = path_item.stem
                                progress.progress(i / total, text=f"[{i}/{total}] {this_doc_id}")
                                try:
                                    res = run_ingestion_quick(
                                        pdf_path=this_uri,
                                        max_sentences=cfg_ing["max_sentences"],
                                        max_text_length=cfg_ing["max_text_length"],
                                        candidate_source=cfg_ui.get("candidate_source", "span"),
                                        coref_backend=cfg_coref.get("engine", "fastcoref"),
                                        coref_device=_ensure_coref_device(cfg_coref),
                                        resolve_text=cfg_coref.get("resolve_text", True),
                                        coref_scope=cfg_coref.get("scope"),
                                        coref_window_sentences=cfg_coref.get("window_sentences"),
                                        coref_window_stride=cfg_coref.get("window_stride"),
                                        coref_shortlist_mode=coref_mode_sidebar,
                                        coref_shortlist_topk=cfg_coref["coref_shortlist_topk"],
                                        coref_trie_tau=cfg_coref["coref_trie_tau"],
                                        coref_cooc_tau=cfg_coref["coref_cooc_tau"],
                                        coref_use_pair_scorer=cfg_coref.get("coref_use_pair_scorer", False),
                                        coref_scorer_threshold=cfg_coref.get("coref_scorer_threshold", 0.25),
                                        coref_pair_scorer=cfg_coref.get("coref_pair_scorer"),
                                        cooc_mode=cooc_mode_val,
                                        cooc_hf_tokenizer=cooc_tok,
                                    )
                                    prod = res.get("production_output") or res

                                    export_production_to_flexiconc(
                                        target_db_path_str,
                                        this_doc_id,
                                        prod,
                                        uri=this_uri,
                                        embedding_models=DEFAULT_EMBEDDING_MODELS,
                                    )

                                    chains = (prod.get("coreference_analysis") or {}).get("chains") or []
                                    if chains:
                                        trie_root, trie_idf, chain_grams = build_ngram_trie(
                                            chains, char_n=4, token_ns=(2, 3)
                                        )

                                        cx2 = open_db(target_db_path_str)
                                        try:
                                            upsert_doc_trie(cx2, this_doc_id, trie_root, trie_idf, chain_grams)

                                            if enable_cooc_sidebar and _HAVE_COOCC_LOCAL:
                                                full_text = prod.get("full_text") or ""
                                                if full_text:
                                                    vocab, rows_c, norms = build_cooc_graph(
                                                        full_text,
                                                        window=5,
                                                        min_count=2,
                                                        topk_neighbors=10,
                                                        mode=cooc_mode_val,
                                                        hf_tokenizer=cooc_tok,
                                                    )
                                                    upsert_doc_cooc(cx2, this_doc_id, vocab, rows_c, norms)
                                            cx2.commit()
                                        finally:
                                            cx2.close()
                                    done += 1
                                    log_container.write(f"✅ {this_doc_id}")
                                except Exception as exc:
                                    errs += 1
                                    log_container.error(f"❌ {this_doc_id}: {exc}")

                            progress.progress(1.0, text="Ingestion complete")
                            log_container.success(
                                f"Ingestion complete. Processed {done} document(s) with {errs} error(s)."
                            )

                            try:
                                cx_final = open_db(target_db_path_str)
                                try:
                                    summary = build_faiss_indices(cx_final)
                                    cx_final.commit()
                                finally:
                                    cx_final.close()
                                if summary:
                                    kinds = ", ".join(sorted(summary.keys()))
                                    log_container.info(
                                        f"FAISS indices refreshed for {len(summary)} model(s): {kinds}."
                                    )
                                else:
                                    log_container.info("No embeddings available for FAISS indices.")
                            except Exception as exc:
                                log_container.warning(f"FAISS index build skipped: {exc}")
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

            _cooc_modes = ["spacy", "hf"]
            current_cooc_mode = str(cfg_coref.get("cooc_mode", "spacy")).lower()
            if current_cooc_mode not in _cooc_modes:
                current_cooc_mode = "spacy"
            cfg_coref["cooc_mode"] = st.selectbox(
                "Co-occurrence backend",
                _cooc_modes,
                index=_cooc_modes.index(current_cooc_mode),
                help="spaCy uses the loaded language model; HF requires a HuggingFace tokenizer.",
            )
            if cfg_coref["cooc_mode"] == "hf":
                cfg_coref["cooc_hf_tokenizer"] = st.text_input(
                    "HF tokenizer (name or path)",
                    value=cfg_coref.get("cooc_hf_tokenizer", ""),
                    help="Leave blank to reuse helper.bert_tokenizer if available.",
                )
            else:
                cfg_coref.setdefault("cooc_hf_tokenizer", cfg_coref.get("cooc_hf_tokenizer", ""))

            # build options
            shortlist_options = ["off", "trie"]
            current_shortlist = cfg_coref.get("coref_shortlist_mode", "trie")
            
            cooc_ready, cooc_msg = cooc_backend_ready(cfg_coref)
            if cooc_ready:
                shortlist_options.extend(["cooc", "both"])
            else:
                if current_shortlist in ("cooc", "both"):
                        current_shortlist = "trie"
                if cooc_msg:
                        st.info(cooc_msg)

            if current_shortlist not in shortlist_options:
                current_shortlist = "trie"

            # dropdown
            cfg_coref["coref_shortlist_mode"] = st.selectbox(
                "Shortlist mode",
                options=shortlist_options,
                index=shortlist_options.index(current_shortlist),
                help="off = no shortlist, trie = token-trie only, cooc = co-occ only, both = union",
                key="coref_shortlist_mode",
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
                cooc_mode_val, cooc_tok, cooc_warn = resolve_cooc_backend(cfg_coref)
                res = run_ingestion_quick(
                    pdf_path=p,
                    max_sentences=st.session_state["config_ingest"]["max_sentences"],
                    max_text_length=st.session_state["config_ingest"]["max_text_length"],
                    # Add the candidate source selection
                    candidate_source=st.session_state["config_ui"]["candidate_source"],
                    coref_backend=cfg_coref.get("engine", "fastcoref"),
                    coref_device = _ensure_coref_device(cfg_coref),
                    coref_scope=cfg_coref.get("scope"),
                    coref_window_sentences=cfg_coref.get("window_sentences"),
                    coref_window_stride=cfg_coref.get("window_stride"),
                    resolve_text=cfg_coref.get("resolve_text"),
                    coref_shortlist_mode=cfg_coref.get("coref_shortlist_mode"),
                    coref_shortlist_topk=cfg_coref.get("coref_shortlist_topk"),
                    coref_trie_tau=cfg_coref.get("coref_trie_tau"),
                    coref_cooc_tau=cfg_coref.get("coref_cooc_tau"),
                    coref_use_pair_scorer=cfg_coref.get("coref_use_pair_scorer"),
                    coref_scorer_threshold=cfg_coref.get("coref_scorer_threshold"),
                    coref_pair_scorer=cfg_coref.get("coref_pair_scorer"),
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
                    cooc_mode=cooc_mode_val,
                    cooc_hf_tokenizer=cooc_tok,
                )
                st.session_state["results"] = res
                st.session_state["last_uri"] = p
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

        def _resolve_path(pth: str) -> str:
            if not pth:
                return pth
            if os.path.isabs(pth):
                return pth
            # join relative to source_root if provided
            return os.path.normpath(os.path.join(source_root or "", pth))

        available_docs = []
        if not Path(db_path).exists():
            st.warning("Set a valid SQLite path.")
        else:
            import pandas as pd
            from flexiconc_adapter import open_db, export_production_to_flexiconc, upsert_doc_trie, upsert_doc_cooc, count_indices, list_index_sizes
            from helper_addons import ensure_documents_table, build_ngram_trie
            try:
                from helper_addons import build_cooc_graph
                _HAVE_COOCC_LOCAL = True
            except Exception:
                _HAVE_COOCC_LOCAL = False

            cx = None
            try:
                # Ensure schema exists / migrate legacy columns if needed
                ensure_documents_table(db_path)
                cx = open_db(db_path)

                # ------ DB health / global coref status ------
                n_trie = count_indices(cx, "trie")
                st.caption(f"Global coref indices (trie): {n_trie}")

                with st.expander("Corpus index / global coref", expanded=(n_trie == 0)):
                    st.write(
                        "Initialize or update **global coreference** indices for documents already present in this DB, "
                        "or build a full corpus from a PDF folder."
                    )
                    cA, cB, cC = st.columns([1, 1, 2])
                    with cA:
                        max_docs = st.number_input("Max docs (0 = all)", min_value=0, value=0, step=1)
                    with cB:
                        enable_cooc = bool(
                            st.session_state.get("sidebar_enable_cooc", False) and _HAVE_COOCC_LOCAL
                        )
                        st.caption(
                            f"Co-occurrence export: {'enabled' if enable_cooc else 'disabled'} (set in sidebar)"
                        )
                    with cC:
                        coref_mode = st.session_state.get("sidebar_coref_mode", "trie")
                        st.caption(
                            f"Coref shortlist mode: {coref_mode} (configured in sidebar)"
                        )

                    cfg_coref = st.session_state["config_coref"]
                    cooc_mode_val, cooc_tok, cooc_warn = resolve_cooc_backend(cfg_coref)
                    if enable_cooc and cooc_warn:
                        st.warning(f"{cooc_warn} Co-occurrence exports will use the spaCy backend.")    
                    # ---- Initialize from existing 'documents' table ----
                    if st.button("Initialize Global Coref (from current DB)"):
                        rows = pd.read_sql_query(
                            "SELECT doc_id, uri FROM documents ORDER BY doc_id", cx
                        ).to_dict("records")
                        if max_docs and max_docs > 0:
                            rows = rows[: int(max_docs)]

                        if not rows:
                            st.warning("No rows in documents table. Use the sidebar ingestion controls to add documents.")
                        else:
                            total = len(rows)
                            prog = st.progress(0.0, text="Starting…")
                            done = skipped = errs = 0
                            for i, r in enumerate(rows, start=1):
                                doc_id = r.get("doc_id")
                                uri = _resolve_path(r.get("uri"))
                                prog.progress(i / total, text=f"[{i}/{total}] {doc_id}")
                                if not uri or not isinstance(uri, str) or not os.path.exists(uri):
                                    skipped += 1
                                    st.warning(f"Skipping {doc_id}: missing or non-existent URI: {uri!r}")
                                    continue
                                try:
                                    # Run fresh ingestion for this file
                                    res = run_ingestion_quick(
                                        pdf_path=uri,
                                        max_sentences=st.session_state["config_ingest"]["max_sentences"],
                                        max_text_length=st.session_state["config_ingest"]["max_text_length"],
                                        candidate_source=st.session_state["config_ui"].get("candidate_source", "span"),
                                        # coref backend / device / resolved text
                                        coref_backend=cfg_coref.get("engine", "fastcoref"),
                                        coref_device = _ensure_coref_device(cfg_coref),
                                        resolve_text=cfg_coref.get("resolve_text", True),
                                        coref_scope=cfg_coref.get("scope"),
                                        coref_window_sentences=cfg_coref.get("window_sentences"),
                                        coref_window_stride=cfg_coref.get("window_stride"),
                                        # shortlist knobs (controls internal trie/cooc shortlisting for span→chain mapping)
                                        coref_shortlist_mode=coref_mode,
                                        coref_shortlist_topk=cfg_coref["coref_shortlist_topk"],
                                        coref_trie_tau=cfg_coref["coref_trie_tau"],
                                        coref_cooc_tau=cfg_coref["coref_cooc_tau"],
                                        coref_use_pair_scorer=cfg_coref.get("coref_use_pair_scorer", False),
                                        coref_scorer_threshold=cfg_coref.get("coref_scorer_threshold", 0.25),
                                        coref_pair_scorer=cfg_coref.get("coref_pair_scorer"),
                                        cooc_mode=cooc_mode_val,
                                        cooc_hf_tokenizer=cooc_tok,
                                    )
                                    prod = res.get("production_output") or res

                                    # 1) One-off schema check / migration (open → print → close)
                                    cx1 = open_db(db_path)  # open_db runs schema migration internally
                                    try:
                                        st.write("documents cols:", [r2[1] for r2 in cx1.execute("PRAGMA table_info(documents)")])
                                        st.write("embeddings cols:", [r2[1] for r2 in cx1.execute("PRAGMA table_info(embeddings)")])
                                        st.write("indices cols:",    [r2[1] for r2 in cx1.execute("PRAGMA table_info(indices)")])
                                    finally:
                                        cx1.close()

                                    # 2) Export the production (uses its own connection internally)
                                    export_production_to_flexiconc(
                                        db_path,
                                        doc_id,
                                        prod,
                                        uri=uri,
                                        embedding_models=DEFAULT_EMBEDDING_MODELS,
                                    )

                                    # 3) Build & persist per-doc indices (reopen a new connection)
                                    chains = (prod.get("coreference_analysis") or {}).get("chains") or []
                                    if chains:
                                        trie_root, trie_idf, chain_grams = build_ngram_trie(chains, char_n=4, token_ns=(2, 3))

                                        cx2 = open_db(db_path)
                                        try:
                                            upsert_doc_trie(cx2, doc_id, trie_root, trie_idf, chain_grams)

                                            # Optional: co-occurrence graph
                                            if enable_cooc and _HAVE_COOCC_LOCAL:
                                                full_text = prod.get("full_text") or ""
                                                if full_text:
                                                    vocab, rows_c, norms = build_cooc_graph(
                                                       full_text,
                                                        window=5,
                                                        min_count=2,
                                                        topk_neighbors=10,
                                                        mode=cooc_mode_val,
                                                        hf_tokenizer=cooc_tok,
                                                    )
                                                    upsert_doc_cooc(cx2, doc_id, vocab, rows_c, norms)
                                            cx2.commit()
                                        finally:
                                            cx2.close()
                                        done += 1
                                    else:
                                        skipped += 1
                                        st.info(f"No chains in analysis for {doc_id}; trie not written.")
                                except Exception as e:
                                    errs += 1
                                    st.exception(e)

                            # 4) Report current trie status (open → read → close)
                            cx3 = open_db(db_path)
                            try:
                                n_now = count_indices(cx3, "trie")
                                st.success(
                                    f"Global coref init complete. "
                                    f"Trie written for {done} doc(s), skipped {skipped}, errors {errs}. "
                                    f"indices(kind='trie') now: {n_now}"
                                )
                                try:
                                    st.caption(f"Sample payload sizes: {list_index_sizes(cx3, 'trie', limit=3)}")
                                except Exception:
                                    pass
                            finally:
                                cx3.close()
                st.info("Use the sidebar controls to ingest folders into FlexiConc or start new databases.")
                # ------ Load corpus rows (documents table preferred; legacy fallback) ------
                try:
                    # Inspect documents schema
                    cols_info = pd.read_sql_query("PRAGMA table_info(documents)", cx)
                    doc_cols = set(cols_info["name"].astype(str))

                    sel_cols = ["doc_id", "uri"]
                    sel_cols.append("created_at" if "created_at" in doc_cols else "NULL AS created_at")
                    if "text_length" in doc_cols:
                        sel_cols.append("text_length")
                    elif "full_text" in doc_cols:
                        sel_cols.append("LENGTH(full_text) AS text_length")
                    else:
                        sel_cols.append("NULL AS text_length")

                    df = pd.read_sql_query(
                        f"SELECT {', '.join(sel_cols)} FROM documents ORDER BY COALESCE(created_at, '') DESC",
                        cx
                    )
                    if not df.empty:
                        st.success(f"Corpus DB ready - {len(df)} documents found")
                        st.dataframe(df, use_container_width=True)
                        available_docs = df.to_dict("records")
                    else:
                        # Legacy fallback
                        st.caption("Using legacy corpus schema (spans_file / files).")
                        tb = None
                        for cand in ("spans_file", "files", "documents"):
                            try:
                                pd.read_sql_query(f"SELECT 1 FROM {cand} LIMIT 1", cx)
                                tb = cand
                                break
                            except Exception:
                                pass
                        if not tb:
                            st.warning("Could not find legacy file table (spans_file/files).")
                        else:
                            meta = pd.read_sql_query(f"PRAGMA table_info({tb})", cx)
                            cols = {r["name"] for _, r in meta.iterrows()}
                            id_col = "id" if "id" in cols else ("rowid" if "rowid" in cols else next(iter(cols)))
                            path_col = next((c for c in ("path","filepath","filename","name") if c in cols), None)
                            if not path_col:
                                st.warning(f"Table {tb} has no path-like column.")
                            else:
                                df2 = pd.read_sql_query(
                                    f"SELECT {id_col} AS doc_id, {path_col} AS uri FROM {tb}",
                                    cx
                                )
                                if df2.empty:
                                    st.info("Legacy table is empty.")
                                else:
                                    # resolve relative paths using source_root (if set)
                                    df2["uri"] = df2["uri"].apply(_resolve_path)
                                    df2["created_at"] = None
                                    df2["text_length"] = None
                                    st.success(f"Found {len(df2)} file entries in {tb}.")
                                    show_cols = [
                                        c for c in df2.columns
                                        if not (c in ("created_at","text_length") and df2[c].isna().all())
                                    ]
                                    st.dataframe(df2[show_cols], use_container_width=True)
                                    available_docs = df2.to_dict("records")
                except Exception as e:
                    st.error(f"Error accessing corpus: {e}")
                finally:
                    if cx is not None:
                        cx.close()
            except Exception as e:
                st.error(f"Error accessing corpus db: {e}")

        # set after DB block completes
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
                                    cooc_mode_val, cooc_tok, cooc_warn = resolve_cooc_backend(cfg_coref)
                                    if cooc_warn and cfg_coref.get("coref_shortlist_mode") in ("cooc", "both"):
                                        st.warning(cooc_warn)
                                    res = run_ingestion_quick(
                                        pdf_path=uri,
                                        max_sentences=cfg_ing["max_sentences"],
                                        max_text_length=cfg_ing["max_text_length"],
                                        coref_backend=cfg_coref.get("engine", "fastcoref"),
                                        coref_device = _ensure_coref_device(cfg_coref),
                                        coref_scope=cfg_coref.get("scope"),
                                        coref_window_sentences=cfg_coref.get("window_sentences"),
                                        coref_window_stride=cfg_coref.get("window_stride"),
                                        resolve_text=cfg_coref.get("resolve_text"),
                                        coref_shortlist_mode=cfg_coref.get("coref_shortlist_mode"),
                                        coref_shortlist_topk=cfg_coref.get("coref_shortlist_topk"),
                                        coref_trie_tau=cfg_coref.get("coref_trie_tau"),
                                        coref_cooc_tau=cfg_coref.get("coref_cooc_tau"),
                                        coref_use_pair_scorer=cfg_coref.get("coref_use_pair_scorer", False),
                                        coref_scorer_threshold=cfg_coref.get("coref_scorer_threshold", 0.25),
                                        coref_pair_scorer=cfg_coref.get("coref_pair_scorer"),
                                        cooc_mode=cooc_mode_val,
                                        cooc_hf_tokenizer=cooc_tok,
                                    )
                                    # attach uri for display; cache rows for single-doc concordance
                                    res = dict(res or {})
                                    res["document_uri"] = uri
                                    st.session_state["results"] = res
                                    st.session_state["current_doc_rows"] = _rows_from_production_local(
                                        _get_production_output(res), uri
                                    )
                                    st.session_state["last_uri"] = uri
                                    prod = _get_production_output(res)
                                    n = len((prod or {}).get("sentence_analyses", []))
                                    if res.get("ok") and n:
                                        st.success(f"✅ Fresh analysis complete. {n} sentences available.")
                                    else:
                                        st.error(f"❌ Analysis failed or empty. {res.get('error','')}")
                                except Exception as e:
                                    st.exception(e)

            # --- Optional: Update corpus with this fresh analysis (explicit actions) ---
            if "results" in st.session_state and st.session_state.get("results"):
                prod = _get_production_output(st.session_state["results"])
                current_uri = st.session_state.get("last_uri")
                if prod and current_uri:
                    c3, c4 = st.columns([1, 1])
                    with c3:
                        save_disabled = not bool(current_uri)
                        if st.button("💾 Save this analysis to corpus", disabled=save_disabled):
                            try:
                                from flexiconc_adapter import export_production_to_flexiconc
                                doc_id_save = Path(current_uri).stem
                                export_production_to_flexiconc(
                                    db_path,
                                    doc_id_save,
                                    prod,
                                    uri=current_uri,
                                    embedding_models=DEFAULT_EMBEDDING_MODELS,
                                )
                                st.success(f"Saved {doc_id_save} into documents/sentences/chains.")
                            except Exception as e:
                                st.error(f"Save failed: {e}")
                    with c4:
                        if st.button("📇 Upsert global coref (this doc)"):
                            try:
                                from flexiconc_adapter import open_db, upsert_doc_trie, count_indices, list_index_sizes
                                from helper_addons import build_ngram_trie
                                cx2 = open_db(db_path)
                                try:
                                    chains = (prod.get("coreference_analysis") or {}).get("chains") or []
                                    if not chains:
                                        st.warning("No coreference chains found in this analysis.")
                                    else:
                                        trie_root, trie_idf, chain_grams = build_ngram_trie(chains, char_n=4, token_ns=(2, 3))
                                        upsert_doc_trie(cx2, Path(current_uri).stem, trie_root, trie_idf, chain_grams)
                                        n_now = count_indices(cx2, "trie")
                                        st.success(f"Trie payload upserted. indices(kind='trie') now: {n_now}")
                                        try:
                                            st.caption(f"Sample payload sizes: {list_index_sizes(cx2, 'trie', limit=3)}")
                                        except Exception:
                                            pass
                                    cx2.commit()
                                finally:
                                    cx2.close()
                            except Exception as e:
                                st.error(f"Upsert failed: {e}")

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
                _handle_phrase_commit()
                
        # --- local coref filter refresh
        prod_now = _get_production_output(st.session_state.get("results"))
        if st.session_state.get("filter_local_coref") and prod_now:
            phrases = (st.session_state.get("query_builder") or {}).get("phrases") or st.session_state.get("query_terms") or []
            filtered = _filter_local_coref_by_phrases(prod_now, phrases, tau_doc=0.15)
            st.session_state["coref_filtered_chains"] = filtered  # [{'chain_id', 'score'}, ...]

        # --- auto-run global coref (Trie)
        if st.session_state.get("auto_run_global_coref"):
            db_path = st.session_state.get("db_path")
            phrases = (st.session_state.get("query_builder") or {}).get("phrases") or st.session_state.get("query_terms") or []
            if db_path and phrases:
                hits = _run_global_coref_from_phrases(
                    db_path,
                    phrases,
                    and_mode=st.session_state.get("global_coref_and_mode", "OR"),
                    tau_trie=st.session_state.get("global_coref_tau_trie", 0.18),
                    topk=st.session_state.get("global_coref_topk", 50),
                )
                st.session_state["global_coref_hits"] = hits

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

                # --- APPLY local filter if enabled and we have results
                if st.session_state.get("filter_local_coref") and st.session_state.get("query_terms"):
                    filt = st.session_state.get("coref_filtered_chains") or []
                    if not filt:
                        # ensure cache if missing
                        _build_local_chain_grams(prod)
                        filt = _filter_local_coref_by_phrases(prod, st.session_state["query_terms"], tau_doc=0.15)
                        st.session_state["coref_filtered_chains"] = filt

                    allowed = {int(r["chain_id"]) for r in filt}
                    # coref_groups is a list of groups, each with 'chain_id' (adapt if your schema differs)
                    coref_groups = [g for g in coref_groups if int(g.get("chain_id")) in allowed]
                    # optional: order by filtered score
                    score_map = {int(r["chain_id"]): r["score"] for r in filt}
                    coref_groups.sort(key=lambda g: score_map.get(int(g.get("chain_id")), 0.0), reverse=True)

                render_coref_panel(coref_groups, prod, st.session_state["config_ui"]["mode"])
                # === Nested, foldable view (chain -> sentence -> mentions) ===
                _ensure_highlight_css()
                st.markdown("#### Coref • Nested view")
                chains = (prod.get("coreference_analysis") or {}).get("chains") or []
                sents  = prod.get("sentence_analyses", []) or []
                by_id  = {int(s.get("sentence_id")): s for s in sents if s.get("sentence_id") is not None}
                
                for ch in chains:
                    cid = ch.get("chain_id")
                    rep = ch.get("representative") or ""
                    ments = ch.get("mentions") or []
                    with st.expander(f"Chain {cid} · rep=“{_preview_text(rep, 60)}” · mentions={len(ments)}", expanded=False):
                        # raw dict (PRAGMA-style)
                        tree("chain (raw)", ch)
                        
                        # bucket mentions by sentence_id (best-effort: use offsets mapping if sentence_id not stored)
                        buckets = {}
                        for m in ments:
                            sid = m.get("sentence_id")
                            if sid is None:
                                # fall back by offset search
                                s0, s1 = m.get("start_char") or m.get("start"), m.get("end_char") or m.get("end")
                                for s in sents:
                                    a, b = s.get("doc_start"), s.get("doc_end")
                                    if a is not None and b is not None and s0 is not None and s1 is not None and a <= s0 and s1 <= b:
                                        sid = s.get("sentence_id"); break
                            buckets.setdefault(sid, []).append(m)

                # render buckets
                for sid, ms in sorted(buckets.items(), key=lambda kv: (10**9 if kv[0] is None else int(kv[0]))):
                    sobj = by_id.get(int(sid)) if sid is not None else None
                    stext = sobj.get("sentence_text") if sobj else ""
                    with st.expander(f"Sentence {sid} · {len(ms)} mention(s) · “{_preview_text(stext, 120)}”", expanded=False):
                        # quick highlight using local spans
                        if sobj:
                            a, b = sobj.get("doc_start"), sobj.get("doc_end")
                            rel = []
                            for m in ms:
                                s0 = m.get("start_char") or m.get("start")
                                s1 = m.get("end_char") or m.get("end")
                                if None not in (s0, s1, a, b) and a <= s0 and s1 <= b:
                                    rel.append((s0 - a, s1 - a))
                            st.markdown(f'<div class="mono">{_highlight_sentence(stext, rel)}</div>', unsafe_allow_html=True)
                        tree("mentions (raw)", ms)

            except Exception as e:
                st.exception(e)  # show error inline to debug
        else:
            st.info("No coreference chains for this sentence.")

with col4:
    with st.container(border=True):
        st.subheader("4) Concordance / Communities & Clusters")
        
        panel_data = st.session_state.get("scico_panel_data")
        _render_chain_hierarchy(panel_data)
        # A) Concordance (corpus mode only, DB-backed)
        if st.session_state["config_ui"]["mode"] == "corpus":
            db_path = st.session_state["db_path"]
            if Path(db_path).exists() and st.session_state["query_terms"]:
                with st.spinner("Querying concordance…"):
                    cfg_corpus = st.session_state["config_corpus"]
                    conc = build_concordance(
                        db_path,
                        st.session_state["query_terms"],
                        and_mode=True,
                        vector_backend=cfg_corpus.get("vector_backend"),
                        use_faiss=cfg_corpus.get("use_faiss", True),
                    )
                st.session_state["corpus_concordance"] = conc
                render_concordance_panel(conc)
                
            else:
                st.session_state["corpus_concordance"] = None
                st.info("Add terms and make sure DB path is valid.")

        # B) SciCo (document & corpus; list only, no viz)
        terms_now = st.session_state.get("query_terms", [])
        terms_clean = [t.strip() for t in terms_now if t and t.strip()]
        run_now = st.button("Run SciCo (using selected terms)")
        auto_requested = st.session_state["config_ui"].get("auto_run_scico") and bool(terms_clean)
        doc_label = _current_doc_label()
        need_auto_run = auto_requested and not (
            panel_data
            and panel_data.get("is_current_doc")
            and panel_data.get("doc_label") == doc_label
            and panel_data.get("terms") == terms_clean
        )

        if run_now or need_auto_run:
            prod = _get_production_output(st.session_state.get("results"))
            precomputed = None
            embedding_provider: Optional[Callable[[List[str]], Any]] = None
            if st.session_state["config_ui"]["mode"] == "corpus":
                conc_res = st.session_state.get("corpus_concordance") or {}
                conc_rows = conc_res.get("rows") or []
                if conc_rows:
                    rows_snapshot = conc_rows
                    precomputed = conc_res.get("embeddings")
                    meta = conc_res.get("meta") or {}
                    vector_backend = meta.get("vector_backend")
                    db_path = st.session_state.get("db_path")
                    if (
                        precomputed is None
                        and vector_backend
                        and db_path
                        and Path(str(db_path)).exists()
                    ):
                        def _embedding_provider(
                            sentences: List[str],
                            rows_snapshot: List[Dict[str, Any]] = rows_snapshot,
                            backend: str = vector_backend,
                            db_path: str = db_path,
                        ) -> List[Any]:
                            try:
                                from flexiconc_adapter import (
                                    open_db,
                                    get_sentence_embedding_cached,
                                )
                            except Exception as exc:
                                raise RuntimeError(
                                    f"embedding fetch unavailable: {exc}"
                                ) from exc

                            conn = open_db(db_path)
                            try:
                                vectors = []
                                for row_obj, sent in zip(rows_snapshot, sentences):
                                    doc_id = row_obj.get("doc_id")
                                    sid = row_obj.get("sentence_id")
                                    if doc_id is None or sid is None:
                                        raise ValueError(
                                            "missing doc_id or sentence_id for embedding lookup"
                                        )
                                    vec = get_sentence_embedding_cached(
                                        conn,
                                        doc_id,
                                        sid,
                                        sent,
                                        model_name=backend,
                                    )
                                    if vec is None:
                                        raise ValueError(
                                            f"no cached embedding for {doc_id}:{sid}"
                                        )
                                    vectors.append(vec)
                                return vectors
                            finally:
                                conn.close()

        embedding_provider = _embedding_providerif not prod:
                warn_panel = {
                    "source": "doc_missing",
                    "doc_label": doc_label,
                    "is_current_doc": True,
                    "error": "No production output available for SciCo.",
                    "terms": terms_clean,
                }
                st.session_state["scico_panel_data"] = warn_panel
                panel_data = warn_panel
                st.warning("No production output available for SciCo.")
            else:
                label = "doc_auto" if (need_auto_run and not run_now) else "doc_manual"
                with st.spinner("Running SciCo…"):
                    panel = _prepare_scico_panel(
                        prod,
                        terms_now,
                        source=label,
                        doc_label=doc_label,
                        include_chains=None,
                        is_current_doc=True,
                        precomputed_embeddings=precomputed,
                        embedding_provider=embedding_provider,
                    )
                st.session_state["scico_panel_data"] = panel
                panel_data = panel

        panel_data = st.session_state.get("scico_panel_data")
        if panel_data and panel_data.get("graph") and panel_data.get("meta") and panel_data.get("is_current_doc"):
            render_clusters_panel(
                panel_data["graph"],
                panel_data["meta"],
                sentence_idx=st.session_state["selected_sentence_idx"],
                summarize_opts={
                    "show_representative": True,
                    "show_xsum": "xsum" in st.session_state["config_scico"]["summary_methods"],
                    "show_presumm": "presumm" in st.session_state["config_scico"]["summary_methods"],
                    },
                )
        elif not terms_clean:
            st.caption("SciCo: add terms and click the button (or enable auto-run).")
        
        # C) Single-doc in-memory concordance (works for both doc mode and fresh corpus analysis)
        _render_single_doc_concordance()

# -------------------- Step 5: Global Coref (Trie, corpus) --------------------
with st.container(border=True):
    st.subheader("5) Global Coref (Trie)")

    db_path = st.session_state.get("db_path")
    phrases = (st.session_state.get("query_builder") or {}).get("phrases") or st.session_state.get("query_terms") or []

    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    with c1:
        st.session_state["global_coref_and_mode"] = st.radio("Mode", ["AND", "OR"],
                                                             index=1 if st.session_state["global_coref_and_mode"]=="OR" else 0,
                                                             horizontal=True, key="gc_and_or")
    with c2:
        st.session_state["global_coref_tau_trie"] = st.slider("τ (trie)", 0.00, 1.00,
                                                              st.session_state["global_coref_tau_trie"], 0.01)
    with c3:
        st.session_state["global_coref_topk"] = st.number_input("Top-K", 1, 500,
                                                                st.session_state["global_coref_topk"], step=5)
    with c4:
        run_gc_btn = st.button("Run Global Coref (Trie)", type="primary", disabled=not(phrases and db_path))

    if run_gc_btn:
        hits = _run_global_coref_from_phrases(
            db_path,
            phrases,
            and_mode=st.session_state["global_coref_and_mode"],
            tau_trie=st.session_state["global_coref_tau_trie"],
            topk=st.session_state["global_coref_topk"],
        )
        st.session_state["global_coref_hits"] = hits

    hits = st.session_state.get("global_coref_hits") or []
    st.caption(f"Results: {len(hits)} chain(s)")

    if not hits:
        st.info("Commit a phrase and run, or enable auto-run in UI settings.")
    else:
        # Simple list with expanders and snippets
        for i, h in enumerate(hits, start=1):
            doc_id = h.get("doc_id")
            chain_id = int(h.get("chain_id"))
            score = float(h.get("score") or h.get("score_trie") or 0.0)
            with st.expander(f"{i}. {doc_id} • chain {chain_id} • score={score:.3f}"):
                _ensure_highlight_css()
                rows = _fetch_chain_sentences_with_mentions(db_path, doc_id, chain_id, limit=6)
                if rows:
                    for r in rows:
                        sid = r["sid"]; stext = r["text"]; spans = r["spans"]
                        with st.expander(f"Sentence {sid} · {len(spans)} mention(s)", expanded=False):
                            st.markdown(f'<div class="mono">{_highlight_sentence(stext, spans)}</div>', unsafe_allow_html=True)
                else:
                    st.caption("No snippet available (empty mentions or missing sentences).")

                cA, cB = st.columns([1,1])
                with cA:
                    if st.button(f"Send to SciCo panel: {doc_id}", key=f"gc_open_{i}"):
                        try:
                            from flexiconc_adapter import load_production_from_flexiconc
                        except Exception as exc:
                            st.error(f"Load failed: {exc}")
                            continue

                        if not db_path:
                            st.error("No corpus database configured for SciCo hand-off.")
                            continue

                        try:
                            prod_loaded = load_production_from_flexiconc(db_path, doc_id)
                        except Exception as exc:
                            st.error(f"Load failed: {exc}")
                            continue

                        if not prod_loaded:
                            st.warning("No stored analysis for this doc.")
                            continue

                        precomputed = None
                        embedding_provider: Optional[Callable[[List[str]], Any]] = None
                        conc_cache = st.session_state.get("corpus_concordance") or {}
                        rows_all = conc_cache.get("rows") or []
                        if rows_all:
                            filtered_rows = [
                                row for row in rows_all if str(row.get("doc_id")) == str(doc_id)
                            ]
                            precomputed_all = conc_cache.get("embeddings")
                            if precomputed_all is not None:
                                precomputed = [
                                    emb
                                    for row, emb in zip(rows_all, precomputed_all)
                                    if str(row.get("doc_id")) == str(doc_id)
                                ]
                            meta = conc_cache.get("meta") or {}
                            vector_backend = meta.get("vector_backend")
                            if (
                                precomputed is None
                                and filtered_rows
                                and vector_backend
                                and Path(str(db_path)).exists()
                            ):
                                def _embedding_provider(
                                    sentences: List[str],
                                    rows_snapshot: List[Dict[str, Any]] = filtered_rows,
                                    backend: str = vector_backend,
                                    db_path: str = db_path,
                                ) -> List[Any]:
                                    try:
                                        from flexiconc_adapter import (
                                            open_db,
                                            get_sentence_embedding_cached,
                                        )
                                    except Exception as exc:
                                        raise RuntimeError(
                                            f"embedding fetch unavailable: {exc}"
                                        ) from exc

                                    conn = open_db(db_path)
                                    try:
                                        vectors = []
                                        for row_obj, sent in zip(rows_snapshot, sentences):
                                            row_doc = row_obj.get("doc_id")
                                            sid = row_obj.get("sentence_id")
                                            if row_doc is None or sid is None:
                                                raise ValueError(
                                                    "missing doc_id or sentence_id for embedding lookup"
                                                )
                                            vec = get_sentence_embedding_cached(
                                                conn,
                                                row_doc,
                                                sid,
                                                sent,
                                                model_name=backend,
                                            )
                                            if vec is None:
                                                raise ValueError(
                                                    f"no cached embedding for {row_doc}:{sid}"
                                                )
                                            vectors.append(vec)
                                        return vectors
                                    finally:
                                        conn.close()

                                embedding_provider = _embedding_provider
                        panel = _prepare_scico_panel(
                                    prod_loaded,
                                    phrases,
                                    source="global_trie",
                                    doc_label=f"{doc_id} · chain {chain_id}",
                                    include_chains=[chain_id],
                                    is_current_doc=False,
                                )
                                st.session_state["scico_panel_data"] = panel
                                if panel.get("error"):
                                    st.error(panel["error"])
                                elif panel.get("warning"):
                                    st.info(panel["warning"])
                                else:
                                    st.success(f"SciCo panel updated for {doc_id} (chain {chain_id}).")
                with cB:
                    if st.button(f"Open doc analysis: {doc_id}", key=f"gc_open_{i}"):
                        try:
                            from flexiconc_adapter import load_production_from_flexiconc
                            prod_loaded = load_production_from_flexiconc(db_path, doc_id)
                            if prod_loaded and prod_loaded.get("sentence_analyses"):
                                st.session_state["results"] = {"production_output": prod_loaded}
                                st.success(f"Loaded analysis for {doc_id}")
                            else:
                                st.warning("No stored analysis for this doc.")
                        except Exception as e:
                            st.error(f"Load failed: {e}")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Built on your existing modules: ENLENS_SpanBert_corefree_prod.py, scico_graph_pipeline.py, helper.py, flexiconc_adapter.py (if present).")

"""
Created on Tue Aug 26 2025
@author: niran
"""
