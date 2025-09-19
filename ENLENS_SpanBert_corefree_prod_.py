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
    # after: sid2span, sent_tree = build_sentence_index(full_text)
    sentences = [full_text[st:en] for _, (st, en) in sorted(sid2span.items())]
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]

    for idx, sent in enumerate(sentences):
        st, en = sid2span[idx]

    # ---- Coref scope switch ----
    backend = kwargs.get("coref_backend", "fastcoref")  # "fastcoref" | "lingmess"
    scope   = kwargs.get("coref_scope", "whole_document")
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
        # spaCy+fastcoref LingMessCoref pipeline
        try:
            import spacy
            from fastcoref import spacy_component  # registers pipe
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

            chains = []
            for cid, cluster in enumerate(doc._.coref_clusters):
                mentions = []
                for sp in cluster:
                    mentions.append({
                        "text": sp.text,
                        "start_char": sp.start_char,
                        "end_char": sp.end_char,
                        "is_pronoun": (sp.root.pos_ == "PRON"),
                        "head_pos": sp.root.pos_
                    })
                representative = max(mentions, key=lambda m: len(m["text"]))["text"] if mentions else ""
                chains.append({"chain_id": cid, "representative": representative, "mentions": mentions})

            if getattr(doc._, "resolved_text", None):
                resolved_text = str(doc._.resolved_text)

            # Add simple antecedent edges with tags
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

        except Exception:
            coref = analyze_full_text_coreferences(full_text) or {}
            chains = normalize_chain_mentions(coref.get("chains", []) or [])

    else:
        # fastcoref default path
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

    # Enrich mentions with sent_id for the UI
    for ch in chains:
        for m in ch.get("mentions", []) or []:
            s0 = int(m.get("start_char", -1))
            if s0 >= 0:
                hits = sent_tree.at(s0)  # payloads: {"sid": int, "start": int, "end": int}
                if hits:
                    m["sent_id"] = hits[0]["sid"]

    # Build interval trees once per chain for quick lookups later
    chain_trees = _build_chain_trees(chains)
    
    char_n   = kwargs.get("trie_char_n", None)                 # None = disable char grams
    token_ns = tuple(kwargs.get("trie_token_ns", (2, 3, 4)))

    ng_index = build_ngram_index(
        chains,
        token_ns=token_ns,
        char_n=char_n,
        build_trie=True,   # fast in-memory path for ENLENS
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
    
    # Shortlist knobs (optionally passed via kwargs)
     
    topk             = int(kwargs.get("coref_shortlist_topk", 5))
    tau_trie         = float(kwargs.get("coref_trie_tau", 0.18))
    tau_cooc         = float(kwargs.get("coref_cooc_tau", 0.18))
    use_scorer       = kwargs.get("coref_use_pair_scorer", False)
    scorer_threshold = float(kwargs.get("coref_scorer_threshold", 0.25))
    pair_scorer      = kwargs.get("coref_pair_scorer", None) if use_scorer else None

    # sentence list (clip to max_sentences)
    sentences = extract_and_filter_sentences(full_text)
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]

    sentence_analyses: List[Dict[str, Any]] = []
    for idx, sent in enumerate(sentences):
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

        # span masking importances (top-K)
        spans = compute_span_importances(sent, target_class=int(pri["label"]), max_span_len=max_span_len)
        spans = sorted(spans, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:top_k_spans]

        span_items: List[Dict[str, Any]] = []
        for rank, sp in enumerate(spans, start=1):
            ls, le = int(sp.get("start", 0)), int(sp.get("end", 0))
            abs_s, abs_e = st + ls, st + le

            # robust query string from normalized variants (sentence-local)
            try:
                variants = normalize_span_for_chaining(sent, ls, le)
                variant_texts = [v[0] for v in variants] or [sp.get("text", sent[ls:le])]
            except Exception:
                variant_texts = [sp.get("text", sent[ls:le])]
            query_text = " ; ".join(dict.fromkeys(variant_texts))[:500]  # cap length

            # shortlist candidate chains
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

            # call mapper with candidates; fallback to old behavior if empty
            coref_info = _map_span_to_chain(
                abs_s, abs_e, chain_trees, chains,
                cand_chain_ids=cand_ids if cand_ids else None,
                scorer=pair_scorer, threshold=(scorer_threshold if pair_scorer else None),
            )

            # keep existing schema; optionally record provenance
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

    try:
        production_output["cluster_analysis"] = prepare_clusters(production_output)
    except Exception:
        production_output["cluster_analysis"] = {"clusters": [], "clusters_dict": {}, "graphs_json": {}}
    
    production_output.setdefault("indices", {})
    production_output["indices"]["coref_ngram"] = ng_index.to_dict()

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
