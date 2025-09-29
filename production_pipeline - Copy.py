"""
production_pipeline.py (normalized orchestrator)
----------------------------------------------
Single entry point that runs the unified production pipeline with a selectable
**candidate_source** ("span" for SpanBERT masking spans, "kpe" for BERT-KPE
keyphrases). Emits the **unified production_output schema** and provides
Streamlit-friendly hooks.

Highlights
- Sentence offsets via `helper.build_sentence_index` (IntervalTree with dict payloads)
- Offset-aware IG via `helper.token_importance_ig`
- Chain normalization via `helper.normalize_chain_mentions` + per-chain IntervalTrees
- Candidate generation pluggable: Span masking (`helper.compute_span_importances`) or KPE
- Cluster analysis via `helper.prepare_clusters`
- Streamlit UI adapter uses `ui_common` for dropdown + overlay
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, Literal
import json

# existing helper: keep using your models and text/coref utilities
from helper import (
    extract_text_from_pdf_robust, preprocess_pdf_text, extract_and_filter_sentences,
    classify_sentence_bert, classify_sentence_similarity, determine_dual_consensus,
    analyze_full_text_coreferences, expand_to_full_phrase, normalize_span_for_chaining,
    nlp, device, SDG_TARGETS
)

# normalized helpers
from helper_addons import (
    Interval, IntervalTree, build_sentence_index, normalize_chain_mentions,
    token_importance_ig, compute_span_importances, prepare_clusters
)

# KPE
from spanbert_kp_extractor import BertKpeExtractor

# UI
from ui_common import streamlit_select_sentence, render_sentence_overlay

# ---------- helpers ----------
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

def _map_abs_span_to_chain(abs_s: int, abs_e: int, chain_trees: Dict[int, IntervalTree],
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

def _classify_dual(sentence: str) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    b_lab, b_conf = classify_sentence_bert(sentence)
    si_lab, si_code, si_conf = classify_sentence_similarity(sentence, SDG_TARGETS)
    cons = determine_dual_consensus(b_lab, b_conf, si_lab, si_conf)
    primary = {"label": int(b_lab), "confidence": float(b_conf)}
    secondary = {"label": (None if si_lab is None else int(si_lab)), "code": si_code, "confidence": float(si_conf)}
    return primary, secondary, cons

# ---------- orchestrator ----------
def run_complete_production_pipeline(pdf_path: str,
                                     candidate_source: Literal["span","kpe"] = "span",
                                     max_sentences: int = 30,
                                     max_span_len: int = 4,
                                     top_k: int = 8,
                                     progress_callback=None) -> Dict[str, Any]:
    if progress_callback: progress_callback("Loading PDF", 0.05)
    raw = extract_text_from_pdf_robust(pdf_path)
    full_text = preprocess_pdf_text(raw)

    if progress_callback: progress_callback("Indexing sentences", 0.12)
    sid2span, sent_tree = build_sentence_index(full_text)
    sentences = extract_and_filter_sentences(full_text)
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]

    if progress_callback: progress_callback("Resolving coreference", 0.22)
    coref = analyze_full_text_coreferences(full_text) or {}
    chains = normalize_chain_mentions(coref.get("chains", []) or [])
    chain_trees = _build_chain_trees(chains)

    if progress_callback: progress_callback("Analyzing sentences", 0.35)
    sentence_analyses: List[Dict[str, Any]] = []

    # optional KPE
    kp_batches = None
    if candidate_source == "kpe":
        kpe = BertKpeExtractor(
            checkpoint_path=BertKpeExtractor.get_default_checkpoint_path(),
            bert_kpe_repo_path=BertKpeExtractor.get_default_repo_path(),
            device=str(device)
        )
        kp_batches = kpe.batch_extract_keyphrases(texts=sentences, top_k=top_k, threshold=0.1)

    for idx, sent in enumerate(sentences):
        if progress_callback: progress_callback(f"Sentence {idx}", 0.35 + 0.6*(idx/len(sentences)))
        st, en = sid2span.get(idx, (full_text.find(sent), full_text.find(sent) + len(sent)))

        pri, sec, cons = _classify_dual(sent)

        toks, scores, offsets = token_importance_ig(sent, int(pri["label"]))
        token_items = [
            {"token": toks[i], "importance": float(scores[i]),
             "start_char": st + int(offsets[i][0]), "end_char": st + int(offsets[i][1])}
            for i in range(len(toks))
        ]

        span_items: List[Dict[str, Any]] = []
        if candidate_source == "span":
            spans = compute_span_importances(sent, target_class=int(pri["label"]), max_span_len=max_span_len)
            spans = sorted(spans, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:top_k]
            for rank, sp in enumerate(spans, start=1):
                ls, le = int(sp.get("start", 0)), int(sp.get("end", 0))
                abs_s, abs_e = st + ls, st + le
                coref_info = _map_abs_span_to_chain(abs_s, abs_e, chain_trees, chains)
                span_items.append({
                    "rank": rank,
                    "original_span": {"text": sp.get("text", sent[ls:le]), "start_char": abs_s, "end_char": abs_e,
                                      "importance": float(sp.get("score", 0.0))},
                    "expanded_phrase": sp.get("text", sent[ls:le]),
                    "coords": [abs_s, abs_e],
                    "coreference_analysis": coref_info
                })
        else:
            raw_kps = kp_batches[idx] if kp_batches else []
            tuples: List[Tuple[str, float]] = []
            for item in raw_kps:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    tuples.append((str(item[0]), float(item[1])))
                elif isinstance(item, dict) and "phrase" in item:
                    tuples.append((str(item["phrase"]), float(item.get("score", 0.0))))
            for rank, (phrase, score) in enumerate(tuples, start=1):
                loc = sent.lower().find(phrase.lower()); 
                if loc < 0: continue
                try:
                    _, (ls, le) = expand_to_full_phrase(sent, loc, loc + len(phrase))
                except Exception:
                    ls, le = loc, loc + len(phrase)
                abs_s, abs_e = st + ls, st + le
                coref_info = _map_abs_span_to_chain(abs_s, abs_e, chain_trees, chains)
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

    if progress_callback: progress_callback("Done", 1.0)
    return production_output

# ---------- Streamlit adapter ----------
def streamlit_run(st, pdf_path: str, candidate_source: Literal["span","kpe"] = "span",
                  max_sentences: int = 30, max_span_len: int = 4, top_k: int = 8):
    prog = st.progress(0.0)
    def _cb(msg, p): prog.progress(min(max(p,0.0),1.0), text=msg)
    out = run_complete_production_pipeline(pdf_path, candidate_source, max_sentences, max_span_len, top_k, _cb)
    sid = streamlit_select_sentence(st, out, source=("span" if candidate_source=="span" else "kp"))
    st.markdown(render_sentence_overlay(out, sid), unsafe_allow_html=True)
    return out

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("pdf", help="Path to PDF file")
    p.add_argument("--candidate_source", choices=["span","kpe"], default="span")
    p.add_argument("--max_sentences", type=int, default=30)
    p.add_argument("--max_span_len", type=int, default=4)
    p.add_argument("--top_k", type=int, default=8)
    args = p.parse_args()
    out = run_complete_production_pipeline(args.pdf, args.candidate_source, args.max_sentences, args.max_span_len, args.top_k)
    print(json.dumps({
        "n_sentences": len(out.get("sentence_analyses", [])),
        "n_chains": out.get("coreference_analysis", {}).get("num_chains", 0),
        "source": args.candidate_source,
        "sample": out.get("sentence_analyses", [])[:1]
    }, indent=2)[:2000])
