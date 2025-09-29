"""
ENLENS_KP_BERT_corefree_prod.py (normalized)
--------------------------------------------
Keyphrase- production pipeline with the **unified production_output schema** and uses the shared
UI helpers from `ui_common.py`.

Highlights:
- Sentence offsets via `helper.build_sentence_index` (IntervalTree with dict payloads).
- Offset-aware IG via `helper.token_importance_ig` → token overlays align to text.
- Chain format tolerance via `helper.normalize_chain_mentions` → per-chain IntervalTrees.
- Keyphrases normalized into `span_analysis` items; chain mapping with probes.
- Optional cluster analysis integrated (same block across all pipelines).
- Standalone UI: uniform dropdown labels + HTML overlay via `ui_common`.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, Literal
import json
import re
 
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# existing helper (models, classifiers, coref…)
from helper import (
    extract_text_from_pdf_robust, preprocess_pdf_text, extract_and_filter_sentences,
    classify_sentence_bert, classify_sentence_similarity, determine_dual_consensus,
    analyze_full_text_coreferences, expand_to_full_phrase, normalize_span_for_chaining,
    nlp, device, SDG_TARGETS
)

# normalized helpers from add-ons
from helper_addons import (
    Interval, IntervalTree, build_sentence_index, normalize_chain_mentions,
    token_importance_ig, prepare_clusters
)

# your existing KPE extractor
from spanbert_kp_extractor import BertKpeExtractor
from ENLENS_SpanBert_corefree_prod import attach_chain_edge_tags
from ui_common import build_sentence_options, render_sentence_overlay

# ---------- chain helpers ----------
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

def _map_phrase_to_chain(sentence_text: str, sent_abs_start: int,
                         phrase: str, chain_trees: Dict[int, IntervalTree],
                         chains: List[Dict[str, Any]]) -> Dict[str, Any]:
    loc = sentence_text.lower().find(phrase.lower())
    if loc < 0: return {"chain_found": False}
    try:
        _, (ls, le) = expand_to_full_phrase(sentence_text, loc, loc + len(phrase))
    except Exception:
        ls, le = loc, loc + len(phrase)
    abs_s, abs_e = sent_abs_start + ls, sent_abs_start + le
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

def run_quick_analysis(pdf_file: str, max_sentences: int = 30, top_k_phrases: int = 5,
                       kpe_threshold: float = 0.1,
                       # Add the missing SDG consensus parameters
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

    sid2span, sent_tree = build_sentence_index(full_text)

    coref = analyze_full_text_coreferences(full_text) or {}
    chains = normalize_chain_mentions(coref.get("chains", []) or [])
    attach_chain_edge_tags(chains)
    chain_trees = _build_chain_trees(chains)

    sentences = extract_and_filter_sentences(full_text)
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]

    kpe = BertKpeExtractor(
        checkpoint_path=BertKpeExtractor.get_default_checkpoint_path(),
        bert_kpe_repo_path=BertKpeExtractor.get_default_repo_path(),
        device=str(device)
    )
    kp_batches = kpe.batch_extract_keyphrases(texts=sentences, top_k=top_k_phrases, threshold=kpe_threshold)

    sentence_analyses: List[Dict[str, Any]] = []
    for idx, sent in enumerate(sentences):
        st, en = sid2span.get(idx, (full_text.find(sent), full_text.find(sent) + len(sent)))

        # Pass the consensus parameters to _classify_dual
        pri, sec, cons = _classify_dual(
            sent,
            agree_threshold=agree_threshold,
            disagree_threshold=disagree_threshold,
            min_confidence=min_confidence
        )

        toks, scores, offsets = token_importance_ig(sent, int(pri["label"]))
        token_items = [
            {"token": toks[i], "importance": float(scores[i]),
             "start_char": st + int(offsets[i][0]), "end_char": st + int(offsets[i][1])}
            for i in range(len(toks))
        ]

        raw_kps = kp_batches[idx] if idx < len(kp_batches) else []
        tuples: List[Tuple[str, float]] = []
        for item in raw_kps:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                tuples.append((str(item[0]), float(item[1])))
            elif isinstance(item, dict) and "phrase" in item:
                tuples.append((str(item["phrase"]), float(item.get("score", 0.0))))

        span_items: List[Dict[str, Any]] = []
        for rank, (phrase, score) in enumerate(tuples, start=1):
            loc = sent.lower().find(phrase.lower())
            if loc < 0: continue
            try:
                _, (ls, le) = expand_to_full_phrase(sent, loc, loc + len(phrase))
            except Exception:
                ls, le = loc, loc + len(phrase)
            abs_s, abs_e = st + ls, st + le
            coref_info = _map_phrase_to_chain(sent, st, phrase, chain_trees, chains)
            span_items.append({
                "rank": rank,
                "original_span": {"text": phrase, "start_char": abs_s, "end_char": abs_e, "importance": float(score)},
                "expanded_phrase": phrase,
                "coords": [abs_s, abs_e],
                "coreference_analysis": coref_info
            })

        sentence_analyses.append({
            "sentence_id": idx,
            "sentence_text": sent,
            "doc_start": st, "doc_end": en,
            "classification": {"label": pri["label"], "score": pri.get("confidence"),
                               "class_id": pri["label"], "consensus": cons, "confidence": pri.get("confidence")},
            "token_analysis": {"tokens": token_items, "max_importance": float(max(scores) if len(scores) else 0.0),
                               "num_tokens": len(token_items)},
            "span_analysis": span_items,
            "metadata": {}
        })

    production_output: Dict[str, Any] = {
        "full_text": full_text,
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
                                     source: Literal["kp","span","auto"] = "kp",
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
    p.add_argument("--top_k_phrases", type=int, default=5)
    args = p.parse_args()
    out = run_quick_analysis(args.pdf, args.max_sentences, args.top_k_phrases)
    print(json.dumps({
        "n_sentences": len(out.get("sentence_analyses", [])),
        "n_chains": out.get("coreference_analysis", {}).get("num_chains", 0),
        "sample": out.get("sentence_analyses", [])[:1]
    }, indent=2)[:2000])
