# -*- coding: utf-8 -*-
# bridge_runners.py
from typing import Dict, Any, List, Tuple, Optional
import importlib
import inspect

from ENLENS_SpanBert_corefree_prod import run_quick_analysis as _run_quick_span

try:
    from ENLENS_KP_BERT_corefree_prod import run_quick_analysis as _run_quick_kpe
except Exception:
    _run_quick_kpe = None

def _unify_sentence_fields(s: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce normalized lists the UI can always rely on:
      - span_terms: list[str]
      - kp_terms:   list[str]
      - token_terms:list[str]
    Leaves original fields intact.
    """
    out = s

    # ---- spans (several possible names across your variants)
    span_blocks = (
        s.get("span_analysis")
        or s.get("spans")
        or s.get("mask_spans")
        or []
    )
    span_terms: List[str] = []
    for it in span_blocks:
        if isinstance(it, dict):
            t = it.get("text") or it.get("span") or it.get("token") or ""
        else:
            t = str(it)
        t = t.strip()
        if t:
            span_terms.append(t)
    out["span_terms"] = span_terms

    # ---- keyphrases (several shapes)
    kp = s.get("keyphrases") or s.get("kpe") or s.get("kp") or []
    if isinstance(kp, dict):
        # common shapes: {"phrases":[{"text":..},..]} or {"topk":[...]}
        kp = kp.get("phrases") or kp.get("topk") or []
    kp_terms: List[str] = []
    for it in kp:
        if isinstance(it, dict):
            t = it.get("text") or it.get("phrase") or ""
        else:
            t = str(it)
        t = t.strip()
        if t:
            kp_terms.append(t)
    out["kp_terms"] = kp_terms

    # ---- tokens (for fallback chips)
    toks = (s.get("token_analysis") or {}).get("tokens") or []
    token_terms: List[str] = []
    for it in toks:
        if isinstance(it, dict):
            t = it.get("token") or it.get("text") or ""
        else:
            t = str(it)
        t = t.strip()
        if t:
            token_terms.append(t)
    out["token_terms"] = token_terms

    return out

def _only_supported_kwargs(fn, **maybe_kwargs):
    """Keep only kwargs present in fn signature."""
    sig = inspect.signature(fn)
    allowed = set(p.name for p in sig.parameters.values())
    return {k: v for k, v in maybe_kwargs.items() if k in allowed}

def _unwrap_production(obj):
    """Accept bare production dict or {'production_output': {...}}."""
    if not isinstance(obj, dict):
        return None
    if "sentence_analyses" in obj and "full_text" in obj:
        return obj
    if "production_output" in obj and isinstance(obj["production_output"], dict):
        return obj["production_output"]
    return None

def run_ingestion_quick(
    pdf_path: str,
    *,
    # top-level size limits
    max_sentences: int | None = None,
    max_text_length: int | None = None,
    # pipeline selection
    candidate_source: str | None = None,  # "span" or "kp"
    # explainability / spans / KPE
    ig_enabled: bool | None = None,
    span_masking_enabled: bool | None = None,
    max_span_len: int | None = None,
    top_k_spans: int | None = None,
    kpe_top_k: int | None = None,
    kpe_threshold: float | None = None,
    # fastcoref knobs (if your pipeline exposes them)
    coref_scope: str | None = None,            # 'whole_document' | 'windowed'
    coref_window_sentences: int | None = None,
    coref_window_stride: int | None = None,
    # sdg / consensus (if exposed)
    agree_threshold: float | None = None,
    disagree_threshold: float | None = None,
    min_confidence: float | None = None,
    # anything else the pipeline might take in the future
    **extra
):
    """UI → bridge → pipeline with safe kwarg filtering and pipeline selection."""
    from pathlib import Path
    
    pdf_path = str(pdf_path)
    if not Path(pdf_path).exists():
        return {"ok": False, "error": f"file not found: {pdf_path}"}

    # Select the appropriate pipeline function
    if candidate_source == "kp" and _run_quick_kpe is not None:
        pipeline_fn = _run_quick_kpe
        # Map some parameter names for KPE pipeline
        raw_kwargs = dict(
            pdf_file=pdf_path,  # Note: KPE uses pdf_file, not pdf_path
            max_sentences=max_sentences,
            max_text_length=max_text_length,
            top_k_phrases=kpe_top_k or top_k_spans,  # KPE uses top_k_phrases
            kpe_threshold=kpe_threshold,
            agree_threshold=agree_threshold,
            disagree_threshold=disagree_threshold,
            min_confidence=min_confidence,
            **extra
        )
    else:
        # Default to SpanBERT pipeline
        pipeline_fn = _run_quick_span
        raw_kwargs = dict(
            pdf_file=pdf_path,
            max_sentences=max_sentences,
            max_text_length=max_text_length,
            max_span_len=max_span_len,
            top_k_spans=top_k_spans,
            agree_threshold=agree_threshold,
            disagree_threshold=disagree_threshold,
            min_confidence=min_confidence,
            **extra
        )
    
    # Drop None values to avoid overriding pipeline defaults
    raw_kwargs = {k: v for k, v in raw_kwargs.items() if v is not None}

    # Keep only arguments that the pipeline really accepts
    pipe_kwargs = _only_supported_kwargs(pipeline_fn, **raw_kwargs)

    # Call selected pipeline
    result = pipeline_fn(**pipe_kwargs)

    # Normalize: always put production dict in "production_output"
    prod = _unwrap_production(result) or result
    if not isinstance(prod, dict):
        return {"ok": False, "error": "Pipeline returned unexpected object."}

    return {"ok": True, "production_output": prod}

def rows_from_production(production_output: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not production_output:
        return []
    rows = []
    for s in (production_output.get("sentence_analyses") or []):
        rows.append(dict(
            sid=s.get("sentence_id"),
            text=s.get("sentence_text"),
            path="",
            start=s.get("doc_start"),
            end=s.get("doc_end"),
        ))
    return rows

def build_scico(rows: List[Dict[str, Any]], selected_terms: List[str], scico_cfg: Dict[str, Any]):
    """
    Wraps scico_graph_pipeline.build_graph_from_selection with the UI dict.
    """
    mod = importlib.import_module("scico_graph_pipeline")
    build = getattr(mod, "build_graph_from_selection")
    ScicoConfig = getattr(mod, "ScicoConfig")

    coherence_opts = dict(
        faiss_topk=scico_cfg["faiss_topk"],
        nprobe=scico_cfg["nprobe"],
        add_lsh=scico_cfg["add_lsh"],
        lsh_threshold=scico_cfg["lsh_threshold"],
        minhash_k=scico_cfg["minhash_k"],
        cheap_len_ratio=scico_cfg["cheap_len_ratio"],
        cheap_jaccard=scico_cfg["cheap_jaccard"],
        use_coherence=scico_cfg["use_coherence"],
        coherence_threshold=scico_cfg["coherence_threshold"],
        max_pairs=scico_cfg["max_pairs"],
    )

    G, meta = build(
        rows,
        selected_terms=selected_terms,
        kmeans_k=scico_cfg["kmeans_k"],
        clustering_method=scico_cfg["clustering"],
        community_on=scico_cfg["community_on"],
        community_method=scico_cfg["community_method"],
        community_weight="prob",
        scico_cfg=ScicoConfig(prob_threshold=scico_cfg["prob_threshold"]),
        add_layout=True,
        use_coherence_shortlist=scico_cfg["use_shortlist"],
        coherence_opts=coherence_opts,
        max_degree=scico_cfg["max_degree"],
        top_edges_per_node=scico_cfg["top_edges_per_node"],
        summarize=scico_cfg["summarize"],
        summarize_on=scico_cfg["summarize_on"],
        summary_methods=scico_cfg["summary_methods"],
        summary_opts=dict(
            num_sentences=scico_cfg["xsum_sentences"],
            sdg_targets=None,
            sdg_top_k=scico_cfg["sdg_topk"],
            cross_encoder_model=scico_cfg["cross_encoder_model"] or None,
            centroid_sim_threshold=scico_cfg["centroid_sim_threshold"],
            centroid_top_n=scico_cfg["centroid_top_n"],
            centroid_store_vector=scico_cfg["centroid_store_vector"],
        ),
    )
    return G, meta


def build_concordance(sqlite_path: str, terms: List[str], and_mode=True):
    """
    Uses flexiconc_adapter if available; otherwise returns [].
    """
    try:
        mod = importlib.import_module("flexiconc_adapter")
        search = getattr(mod, "query_concordance", None)
        if search is None:
            return []
        return search(sqlite_path, terms, mode="AND" if and_mode else "OR", limit=500)
    except Exception:
        return []


def pick_sentence_coref_groups(production_output: Dict[str, Any], sent_idx: int):
    """
    From your production output structure: collect chains that include the sentence,
    return {chain_id: [ {sid, text, score?}, ... ] }
    """
    chains = (production_output.get("coreference_analysis")
              or {}).get("chains") or []
    sents = production_output.get("sentence_analyses") or []
    if not chains or sent_idx is None or sent_idx >= len(sents):
        return {}

    # naive mapping: if any mention in the chain falls into this sentence span, we include it
    out = {}
    target = sents[sent_idx]
    t0, t1 = target.get("doc_start"), target.get("doc_end")
    if t0 is None or t1 is None:
        return {}

    for ci, chain in enumerate(chains):
        members = []
        for m in chain.get("mentions", []):
            s = m.get("sent_id")
            start = m.get("start_char")
            end = m.get("end_char")
            score = m.get("score", None)
            if s is None or s >= len(sents):
                continue
            # include all mentions in chain; we’ll rely on UI list for display
            members.append(dict(
                sid=s,
                text=sents[s].get("sentence_text", ""),
                score=score
            ))
        if members:
            out[ci] = members
    return out


def compute_sentence_clusters_for_doc(G, meta, sent_idx: int):
    # Placeholder if you later want to compute detailed cluster membership lists
    return {}


"""
Created on Mon Aug 25 15:03:33 2025

@author: niran
"""
