"""
ENLENS_SpanBert_corefree_prod.py (normalized)
--------------------------------------------
SpanBERT production with **unified production_output schema**
and using shared UI helpers from `ui_common.py`.

Highlights:
- Sentence offsets via `helper.build_sentence_index` (IntervalTree with dict payloads).
- Offset-aware IG via `helper.token_importance_ig` → token overlays align to text.
- Span masking importance via `helper.compute_span_importances` (or equivalent) → spans.
- Chain format tolerance via `helper.normalize_chain_mentions` → per-chain IntervalTrees.
- Cluster analysis integrated (same block across all pipelines).
- Standalone UI: uniform dropdown labels + HTML overlay via `ui_common`.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, Literal
import json
import re

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# use your existing helper for models + text IO + classifiers + coref
from helper import (
    extract_text_from_pdf_robust, preprocess_pdf_text, extract_and_filter_sentences,
    classify_sentence_bert, classify_sentence_similarity, determine_dual_consensus,
    analyze_full_text_coreferences, expand_to_full_phrase, normalize_span_for_chaining,
    nlp, device, SDG_TARGETS, _fastcoref_in_windows
)

# import normalized helpers from add-ons
from helper_addons import (
    Interval, IntervalTree, build_sentence_index, normalize_chain_mentions,
    token_importance_ig, compute_span_importances, prepare_clusters, build_ngram_trie, shortlist_by_trie,
    build_cooc_graph, shortlist_by_cooc, fuse_shortlists, build_spacy_trie_with_idf, build_ngram_index
)

from ui_common import build_sentence_options, render_sentence_overlay

from global_coref_helper import (
    build_global_superroot, global_coref_query
)

from span_chains import _map_span_to_chain, _build_chain_trees
from lingmess_coref import ensure_fastcoref_component, make_lingmess_nlp, run_lingmess_coref

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in some envs
    torch = None

def _available_devices() -> List[str]:
    devices = ["cpu"]
    if torch is not None:
        try:
            if torch.cuda.is_available():
                count = torch.cuda.device_count() or 1
                devices.extend([f"cuda:{i}" for i in range(count)])
        except Exception:
            pass
    return devices


def _normalize_device(requested: Optional[str]) -> str:
    opts = _available_devices()
    default = next((opt for opt in opts if opt.startswith("cuda")), opts[0])
    if requested is None:
        return default
    value = str(requested).strip()
    if not value or value.lower() in {"auto", "default"}:
        return default
    lowered = value.lower()
    if lowered == "cuda":
        return default if default.startswith("cuda") else "cpu"
    for opt in opts:
        if lowered == opt.lower():
            return opt
    if lowered.startswith("cuda:") and any(opt.startswith("cuda") for opt in opts):
        return default
    return default

_PRONOUN_TOKENS = {
    "he", "she", "it", "they", "him", "her", "them", "his", "hers", "its",
    "their", "theirs", "we", "us", "i", "you", "yours", "mine", "ours",
    "himself", "herself", "itself", "ourselves", "themselves", "yourself",
    "yourselves", "myself"
}


def _normalize_pronoun_token(text: str) -> str:
    token = re.sub(r"[^a-z]", "", str(text or "").lower())
    return token


def _is_pronoun_like(text: str) -> bool:
    return _normalize_pronoun_token(text) in _PRONOUN_TOKENS


def _tag_pair(ant_txt, ana_txt, ant_is_pron, ana_is_pron):
    if ant_is_pron and ana_is_pron:
        return "PRON-PRON-C"
    if ant_txt == ana_txt:
        return "MATCH"
    if ant_txt in ana_txt or ana_txt in ant_txt:
        return "CONTAINS"
    if (not ant_is_pron) and ana_is_pron:
        return "ENT-PRON"
    return "OTHER"


def attach_chain_edge_tags(chains: List[Dict[str, Any]]) -> None:
    for ch in chains or []:
        mentions = ch.get("mentions", []) or []
        if len(mentions) < 2:
            ch.setdefault("edges", [])
            continue

        existing_edges = ch.get("edges") or []
        if existing_edges and any("tag" in e for e in existing_edges if isinstance(e, dict)):
            # Already tagged (e.g., LingMess rich output)
            continue

        for m in mentions:
            if not isinstance(m, dict):
                continue
            if ("is_pronoun" not in m) or (m.get("is_pronoun") is None):
                m["is_pronoun"] = _is_pronoun_like(m.get("text", ""))

        new_edges: List[Dict[str, Any]] = []
        last_non_pron: Optional[int] = None
        for idx in range(1, len(mentions)):
            ant_idx = last_non_pron if last_non_pron is not None else (idx - 1)
            if ant_idx < 0 or ant_idx >= len(mentions):
                ant_idx = idx - 1
            ant = mentions[ant_idx]
            ana = mentions[idx]
            tag = _tag_pair(
                ant.get("text", ""),
                ana.get("text", ""),
                bool(ant.get("is_pronoun")),
                bool(ana.get("is_pronoun")),
            )
            new_edges.append({"antecedent": int(ant_idx), "anaphor": int(idx), "tag": tag})
            if not bool(ana.get("is_pronoun")):
                last_non_pron = idx

        ch["edges"] = new_edges

# ---------- main ----------
def _classify_dual(sentence: str, 
                   agree_threshold: float = 0.1,
                   disagree_threshold: float = 0.2, 
                   min_confidence: float = 0.5) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    b_lab, b_conf = classify_sentence_bert(sentence)
    si_lab, si_code, si_conf = classify_sentence_similarity(sentence, SDG_TARGETS)
    cons = determine_dual_consensus(
        b_lab, b_conf, si_lab, si_conf,
        agree_thresh=agree_threshold,
        disagree_thresh=disagree_threshold, 
        min_conf=min_confidence
    )
    primary = {"label": int(b_lab), "confidence": float(b_conf)}
    secondary = {"label": (None if si_lab is None else int(si_lab)), "code": si_code, "confidence": float(si_conf)}
    return primary, secondary, cons

def run_quick_analysis(
    pdf_file: str,
    max_sentences: int = 30,
    max_span_len: int = 4,
    top_k_spans: int = 8,
    # SDG consensus parameters
    agree_threshold: float = 0.1,
    disagree_threshold: float = 0.2,
    min_confidence: float = 0.5,
    # Accept other kwargs that might be passed by the bridge
    **kwargs
) -> Dict[str, Any]:
    # Extract max_text_length if provided
    max_text_length = kwargs.get("max_text_length")

    raw = extract_text_from_pdf_robust(pdf_file)
    if max_text_length is not None:
        full_text = preprocess_pdf_text(raw, max_length=int(max_text_length))
    else:
        full_text = preprocess_pdf_text(raw)

    # sentence index + interval tree (absolute offsets)
    sid2span, sent_tree = build_sentence_index(full_text)

    # ---- Coref scope/backend switches ----
    sentence_analyses = None,
    backend = kwargs.get("coref_backend", "fastcoref")  # "fastcoref" | "lingmess"
    scope   = kwargs.get("coref_scope", "whole_document")
    resolved_text = None
    
    
    # ---------------- LingMess backend ----------------
    if backend == "lingmess":
        chains = []
        try:
            # Use helper for robust parsing + edges
            ensure_fastcoref_component()
            
            coref_device = _normalize_device(kwargs.get("coref_device"))
            want_resolved = bool(kwargs.get("resolve_text", True))

            nlp, resolver = make_lingmess_nlp(device=coref_device, eager_attention=True)

            # sentence_analyses not available yet; we’ll add sent_ids below from sent_tree anyway
            lm = run_lingmess_coref(
                full_text,
                nlp,
                resolver=resolver,
                sentence_analyses=None
            )
            chains = lm.get("chains", []) or []

            # Normalize edges to your app’s schema: {antecedent, anaphor, tag}
            for ch in chains:
                conv = []
                for e in ch.get("edges") or []:
                    i = e.get("i", e.get("antecedent", -1))
                    j = e.get("j", e.get("anaphor", -1))
                    rel = e.get("relation") or e.get("tag") or "OTHER"
                    try:
                        i = int(i); j = int(j)
                    except Exception:
                        continue
                    if i >= 0 and j >= 0:
                        conv.append({"antecedent": i, "anaphor": j, "tag": str(rel)})
                ch["edges"] = conv

            # Optionally capture resolved text
            if want_resolved and lm.get("resolved_text"):
                resolved_text = lm["resolved_text"]

        except Exception:
            # Fallback to your legacy analyzer
            coref = analyze_full_text_coreferences(full_text) or {}
            chains = normalize_chain_mentions(coref.get("chains", []) or [])
        
        attach_chain_edge_tags(chains)
    # ---------------- fastcoref default backend ----------------
    else:
        if scope == "windowed":
            clust = _fastcoref_in_windows(
                full_text,
                k_sentences=int(kwargs.get("coref_window_sentences", 3)),
                stride=int(kwargs.get("coref_window_stride", 2)),
            )
            chains = []
            for cid, cluster in enumerate(clust):
                mentions = [{"text": full_text[s:e], "start_char": s, "end_char": e} for (s, e) in cluster]
                if mentions:
                    representative = max(mentions, key=lambda m: len(m["text"]))["text"]
                    chains.append({"chain_id": cid, "representative": representative, "mentions": mentions})
        else:
            coref = analyze_full_text_coreferences(full_text) or {}
            chains = normalize_chain_mentions(coref.get("chains", []) or [])
        attach_chain_edge_tags(chains)    
    # ------------- Enrich mentions with sent_id via interval tree -------------
    for ch in chains:
        for m in ch.get("mentions", []) or []:
            s0 = int(m.get("start_char", -1))
            if s0 >= 0:
                hits = sent_tree.at(s0)
            if hits:
                m["sent_id"] = hits[0]["sid"]
    # ------------- Build interval trees per chain (for fast mapping) ----------
    chain_trees = _build_chain_trees(chains)

    # ------------- Build indices for shortlist (trie/cooc) --------------------
    from helper_addons import build_ngram_index, build_cooc_graph

    char_n   = kwargs.get("trie_char_n", None)                # None disables char-grams
    token_ns = tuple(kwargs.get("trie_token_ns", (2, 3, 4)))

    ng_index = build_ngram_index(
        chains,
        token_ns=token_ns,
        char_n=char_n,
        build_trie=True,
    )

    shortlist_mode = str(kwargs.get("coref_shortlist_mode", "off")).lower()  # "off"|"trie"|"cooc"|"both"
    if shortlist_mode in ("cooc", "both"):
        vocab, rows, row_norms = build_cooc_graph(
            full_text,
            window=int(kwargs.get("cooc_window", 5)),
            min_count=int(kwargs.get("cooc_min_count", 3)),
            topk_neighbors=int(kwargs.get("cooc_topk_neighbors", 20)),
            mode=kwargs.get("cooc_mode", "hf"),
            hf_tokenizer=kwargs.get("cooc_hf_tokenizer"),
            cooc_impl=kwargs.get("cooc_impl", "fast"),
            cooc_max_tokens=int(kwargs.get("cooc_max_tokens", 50000)),
            cache_key=hash(full_text) % (10**9),
        )
    else:
        vocab, rows, row_norms = {}, [], {}

    # Shortlist knobs
    topk             = int(kwargs.get("coref_shortlist_topk", 5))
    tau_trie         = float(kwargs.get("coref_trie_tau", 0.18))
    tau_cooc         = float(kwargs.get("coref_cooc_tau", 0.18))
    use_scorer       = kwargs.get("coref_use_pair_scorer", False)
    scorer_threshold = float(kwargs.get("coref_scorer_threshold", 0.25))
    pair_scorer      = kwargs.get("coref_pair_scorer", None) if use_scorer else None

    # ---------------- Sentence list (clip to max_sentences) -------------------
    sentences = extract_and_filter_sentences(full_text)
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]

    sentence_analyses: List[Dict[str, Any]] = []
    for idx, sent in enumerate(sentences):
        # robust absolute bounds from sid2span; fallback if missing
        st, en = sid2span.get(idx, (full_text.find(sent), full_text.find(sent) + len(sent)))

        # Dual classifier (with consensus knobs)
        pri, sec, cons = _classify_dual(
            sent,
            agree_threshold=agree_threshold,
            disagree_threshold=disagree_threshold,
            min_confidence=min_confidence,
        )

        # IG token importances (absolute offsets)
        toks, scores, offsets = token_importance_ig(sent, int(pri["label"]))
        token_items = [
            {"token": toks[i], "importance": float(scores[i]),
             "start_char": st + int(offsets[i][0]), "end_char": st + int(offsets[i][1])}
            for i in range(len(toks))
        ]

        # Span masking importances (top-K)
        spans = compute_span_importances(sent, target_class=int(pri["label"]), max_span_len=max_span_len)
        spans = sorted(spans, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:top_k_spans]

        span_items: List[Dict[str, Any]] = []
        for rank, sp in enumerate(spans, start=1):
            ls, le = int(sp.get("start", 0)), int(sp.get("end", 0))
            abs_s, abs_e = st + ls, st + le

            # Build query text from normalized variants
            try:
                variants = normalize_span_for_chaining(sent, ls, le)
                variant_texts = [v[0] for v in variants] or [sp.get("text", sent[ls:le])]
            except Exception:
                variant_texts = [sp.get("text", sent[ls:le])]
            query_text = " ; ".join(dict.fromkeys(variant_texts))[:500]

            # Shortlist candidate chains
            cand_ids: List[int] = []
            triers: List[Tuple[int, float]] = []
            coocs:  List[Tuple[int, float]] = []

            if shortlist_mode in ("trie", "both"):
                triers = ng_index.shortlist(query_text, topk=topk, tau=tau_trie)

            if shortlist_mode in ("cooc", "both"):
                coocs = shortlist_by_cooc(
                    query_text,
                    chains=chains,
                    vocab=vocab, rows=rows, row_norms=row_norms,
                    topk=topk, tau=tau_cooc,
                )

            if shortlist_mode == "trie":
                cand_ids = [cid for cid, _ in triers]
            elif shortlist_mode == "cooc":
                cand_ids = [cid for cid, _ in coocs]
            elif shortlist_mode == "both":
                cand_ids = fuse_shortlists(triers, coocs, wA=0.6, wB=0.4, topk=topk)
            else:  # "off"
                cand_ids = []

            # Map span → chain (with optional scorer)
            coref_info = _map_span_to_chain(
                abs_s, abs_e, chain_trees, chains,
                cand_chain_ids=cand_ids if cand_ids else None,
                scorer=pair_scorer, threshold=(scorer_threshold if pair_scorer else None),
            )

            span_items.append({
                "rank": rank,
                "original_span": {"text": sp.get("text", sent[ls:le]),
                                  "start_char": abs_s, "end_char": abs_e,
                                  "importance": float(sp.get("score", 0.0))},
                "expanded_phrase": sp.get("text", sent[ls:le]),
                "coords": [abs_s, abs_e],
                "coreference_analysis": (
                    {**coref_info, "decision": ("shortlist+scorer" if pair_scorer else "shortlist")}
                    if coref_info.get("chain_found")
                    else {"chain_found": False, "decision": "none"}
                )
            })

        sentence_analyses.append({
            "sentence_id": idx,
            "sentence_text": sent,
            "doc_start": st, "doc_end": en,
            "classification": {"label": pri["label"], "score": pri.get("confidence"),
                               "class_id": pri["label"], "consensus": cons, "confidence": pri.get("confidence")},
            "token_analysis": {"tokens": token_items,
                               "max_importance": float(max(scores) if len(scores) else 0.0),
                               "num_tokens": len(token_items)},
            "span_analysis": span_items,
            "metadata": {}
        })

    
    production_output: Dict[str, Any] = {
        "full_text": full_text,
        "resolved_text": resolved_text,
        "document_analysis": {},
        "coreference_analysis": {"num_chains": len(chains), "chains": chains},
        "sentence_analyses": sentence_analyses
    }
    
    production_output.setdefault("coreference_analysis", {})
    production_output["coreference_analysis"]["backend_used"] = backend
    production_output["coreference_analysis"]["num_chains"] = len(chains)

    try:
        production_output["cluster_analysis"] = prepare_clusters(production_output)
    except Exception:
        production_output["cluster_analysis"] = {"clusters": [], "clusters_dict": {}, "graphs_json": {}}

    # Persist ngram index snapshot (for debugging / UI)
    production_output.setdefault("indices", {})
    try:
        production_output["indices"]["coref_ngram"] = ng_index.to_dict()
    except Exception:
        production_output["indices"]["coref_ngram"] = {}

    return production_output
 
# ---------- minimal ipywidgets UI ----------
def create_interactive_visualization(production_output: Dict[str, Any],
                                     source: Literal["kp","span","auto"] = "span",
                                     return_widgets: bool = False):
    labels, indices = build_sentence_options(production_output, source=source)
    selector = widgets.Dropdown(options=[("Select a sentence.", None)] + list(zip(labels, indices)),
                                description="Sentence:")
    out = widgets.Output()

    def _on_change(change):
        if change["name"] == "value" and change["new"] is not None:
            sid = int(change["new"]) if not isinstance(change["new"], tuple) else int(change["new"][1])
            with out:
                clear_output()
                html = render_sentence_overlay(production_output, sid, highlight_coref=True, box_spans=True)
                display(HTML(html))

    selector.observe(_on_change, names="value")
    ui = widgets.VBox([selector, out])
    if return_widgets:
        return ui, {"selector": selector, "output": out}
    display(ui)
    return ui

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("pdf", help="Path to PDF file")
    p.add_argument("--max_sentences", type=int, default=30)
    p.add_argument("--max_span_len", type=int, default=4)
    p.add_argument("--top_k_spans", type=int, default=8)
    args = p.parse_args()
    out = run_quick_analysis(args.pdf, args.max_sentences, args.max_span_len, args.top_k_spans)
    print(json.dumps({
        "n_sentences": len(out.get("sentence_analyses", [])),
        "n_chains": out.get("coreference_analysis", {}).get("num_chains", 0),
        "sample": out.get("sentence_analyses", [])[:1]
    }, indent=2)[:2000])
