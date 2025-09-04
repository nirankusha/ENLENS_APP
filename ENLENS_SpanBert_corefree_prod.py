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

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# use your existing helper for models + text IO + classifiers + coref
from helper import (
    extract_text_from_pdf_robust, preprocess_pdf_text, extract_and_filter_sentences,
    classify_sentence_bert, classify_sentence_similarity, determine_dual_consensus,
    analyze_full_text_coreferences, expand_to_full_phrase, normalize_span_for_chaining,
    nlp, device, SDG_TARGETS
)

# import normalized helpers from add-ons
from helper_addons import (
    Interval, IntervalTree, build_sentence_index, normalize_chain_mentions,
    token_importance_ig, compute_span_importances, prepare_clusters
)

from ui_common import build_sentence_options, render_sentence_overlay

# ---------- span/chain helpers ----------
def _build_chain_trees(chains: List[Dict[str, Any]]) -> Dict[int, IntervalTree]:
    trees: Dict[int, IntervalTree] = {}
    for ch in chains:
        cid = int(ch.get("chain_id", -1))
        nodes: List[Interval] = []
        for m in ch.get("mentions", []) or []:
            m0, m1 = int(m.get("start_char", -1)), int(m.get("end_char", -1))
            if 0 <= m0 < m1:
                nodes.append(Interval(m0, m1, m))
        trees[cid] = IntervalTree(nodes)
    return trees

def _map_span_to_chain(abs_s: int, abs_e: int, chain_trees: Dict[int, IntervalTree],
                       chains: List[Dict[str, Any]]) -> Dict[str, Any]:
    probes = [abs_s, (abs_s + abs_e)//2, max(abs_s, abs_e - 1)]
    for ch in chains:
        tree = chain_trees.get(int(ch.get("chain_id", -1)))
        if not tree: continue
        if any(tree.at(p) for p in probes):
            related = []
            for m in ch.get("mentions", [])[:5]:
                m0, m1 = int(m.get("start_char", -1)), int(m.get("end_char", -1))
                if not (m0 == abs_s and m1 == abs_e):
                    related.append({"text": m.get("text", ""), "coords": [m0, m1]})
            return {"chain_found": True, "chain_id": int(ch.get("chain_id", -1)),
                    "representative": ch.get("representative", ""), "related_mentions": related}
    return {"chain_found": False}

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

def run_quick_analysis(pdf_file: str, max_sentences: int = 30,
                       max_span_len: int = 4, top_k_spans: int = 8,
                       # SDG consensus parameters
                       agree_threshold: float = 0.1,
                       disagree_threshold: float = 0.2,
                       min_confidence: float = 0.5,
                       # Accept other kwargs that might be passed by the bridge
                       **kwargs) -> Dict[str, Any]:
    # Extract max_text_length if provided
    max_text_length = kwargs.get("max_text_length")

    raw = extract_text_from_pdf_robust(pdf_file)
    if max_text_length is not None:
        full_text = preprocess_pdf_text(raw, max_length=int(max_text_length))
    else:
        full_text = preprocess_pdf_text(raw)

    # sentence index + interval tree
    sid2span, sent_tree = build_sentence_index(full_text)

    # ---- Coref scope switch ----
    backend = kwargs.get("coref_backend", "fastcoref")  # "fastcoref" | "lingmess"
    scope   = kwargs.get("coref_scope", "whole_document")  # keep your windowed option if you like
    resolved_text = None

    def _tag_pair(ant_txt, ana_txt, ant_is_pron, ana_is_pron):
       if ant_is_pron and ana_is_pron:
           return "PRON-PRON-C"   # within-chain pron↔pron co-ref
       if ant_txt == ana_txt:
           return "MATCH"
       if ant_txt in ana_txt or ana_txt in ant_txt:
           return "CONTAINS"
       if (not ant_is_pron) and ana_is_pron:
           return "ENT-PRON"
       return "OTHER"

    if backend == "lingmess":
       # Preferred: spaCy component → doc._.coref_clusters & doc._.resolved_text
       try:
           import spacy
           from fastcoref import spacy_component  # register "fastcoref" pipe
           # Use your existing nlp if available; else create a blank pipeline
           try:
               _nlp = nlp  # from helper
           except Exception:
               _nlp = spacy.blank("en")
           if "fastcoref" not in _nlp.pipe_names:
               _nlp.add_pipe("fastcoref", config={
                   "model_architecture": "LingMessCoref",
                   "device": kwargs.get("coref_device", "cuda:0"),
                   "resolve_text": kwargs.get("resolve_text", True)
               })
           doc = _nlp(full_text)

           # Build chains
           chains = []
           for cid, cluster in enumerate(doc._.coref_clusters):
               # cluster is a list of Span objects
               mentions = []
               for sp in cluster:
                   mentions.append({
                       "text": sp.text,
                       "start_char": sp.start_char,
                       "end_char": sp.end_char,
                       # enrich later with sent_id; POS/pronoun flags:
                       "is_pronoun": (sp.root.pos_ == "PRON"),
                       "head_pos": sp.root.pos_
                   })
               # representative = longest mention by char length
               representative = max(mentions, key=lambda m: len(m["text"]))["text"] if mentions else ""
               chains.append({"chain_id": cid, "representative": representative, "mentions": mentions})

           # Optional resolved text (if resolve_text=True was given)
           if getattr(doc._, "resolved_text", None):
               resolved_text = str(doc._.resolved_text)

           # Add antecedent edges + 6-way tags (approx.)
           # Heuristic antecedent = closest previous non-pronoun in cluster; else previous mention
           for ch in chains:
               m = ch["mentions"]
               edges = []
               last_non_pron = None
               for i in range(1, len(m)):
                   ant_idx = last_non_pron if last_non_pron is not None else (i - 1)
                   ant = m[ant_idx]; ana = m[i]
                   tag = _tag_pair(ant["text"], ana["text"], ant.get("is_pronoun", False), ana.get("is_pronoun", False))
                   edges.append({"antecedent": ant_idx, "anaphor": i, "tag": tag})
                   if not ana.get("is_pronoun", False):
                       last_non_pron = i
               ch["edges"] = edges

       except Exception as e:
           # Fallback to default whole-document fastcoref
           coref = analyze_full_text_coreferences(full_text) or {}
           chains = normalize_chain_mentions(coref.get("chains", []) or [])

    else:
       # Existing fastcoref path (your current behavior)
       if scope == "windowed":
           from helper import _fastcoref_in_windows
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
    # Enrich mentions with sent_id for the UI (widget 3)
    for ch in chains:
        for m in ch.get("mentions", []) or []:
            s0 = int(m.get("start_char", -1))
            if s0 >= 0:
                hits = sent_tree.at(s0)  # payloads: {"sid": int, "start": int, "end": int}
                if hits:
                    m["sent_id"] = hits[0]["sid"]

    # Build interval trees once per chain for quick lookups later
    chain_trees = _build_chain_trees(chains)

    # sentence list (clip to max_sentences)
    sentences = extract_and_filter_sentences(full_text)
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]

    sentence_analyses: List[Dict[str, Any]] = []
    for idx, sent in enumerate(sentences):
        st, en = sid2span.get(idx, (full_text.find(sent), full_text.find(sent) + len(sent)))

        # Dual classifier (with your consensus knobs)
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

        # span masking importances (top-K)
        spans = compute_span_importances(sent, target_class=int(pri["label"]), max_span_len=max_span_len)
        spans = sorted(spans, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:top_k_spans]

        span_items: List[Dict[str, Any]] = []
        for rank, sp in enumerate(spans, start=1):
            ls, le = int(sp.get("start", 0)), int(sp.get("end", 0))
            abs_s, abs_e = st + ls, st + le
            coref_info = _map_span_to_chain(abs_s, abs_e, chain_trees, chains)
            span_items.append({
                "rank": rank,
                "original_span": {"text": sp.get("text", sent[ls:le]),
                                  "start_char": abs_s, "end_char": abs_e,
                                  "importance": float(sp.get("score", 0.0))},
                "expanded_phrase": sp.get("text", sent[ls:le]),
                "coords": [abs_s, abs_e],
                "coreference_analysis": coref_info
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

    try:
        production_output["cluster_analysis"] = prepare_clusters(production_output)
    except Exception:
        production_output["cluster_analysis"] = {"clusters": [], "clusters_dict": {}, "graphs_json": {}}

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
