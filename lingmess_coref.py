# -*- coding: utf-8 -*-
# lingmess_coref.py
# Utilities to run LingMess (fastcoref) and emit chains + rich "edges" for UI.

from __future__ import annotations
import os, re
from typing import List, Dict, Any, Optional, Tuple

def _safe_imports(device: str, eager_attn: bool):
    # optional: stabilize attention kernel on some CUDA stacks
    if eager_attn:
        os.environ.setdefault("TRANSFORMERS_ATTN_IMPLEMENTATION", "eager")
        try:
            import torch
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
        except Exception:
            pass
    import spacy                           # noqa
    from fastcoref.spacy_component import spacy_component  # noqa (registers "fastcoref")
    return spacy

LINGMESS_CATEGORIES = {
    -1: "NO_RELATION",
     0: "NOT_COREF",
     1: "PRON_PRON_C",
     2: "PRON_PRON_NC",
     3: "ENT_PRON",
     4: "MATCH",
     5: "CONTAINS",
     6: "OTHER",
}

_PRON_SET = {"he","she","it","they","him","her","them","his","hers","its","their","theirs"}

def make_lingmess_nlp(device: str = "cpu", eager_attention: bool = True):
    """
    Returns (nlp, resolver). Safe to call once and cache.
    """
    spacy = _safe_imports(device, eager_attention)
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(
        "fastcoref",
        config={
            "model_architecture": "LingMessCoref",
            "model_path": "biu-nlp/lingmess-coref",
            "device": device,
        },
    )
    resolver = nlp.get_pipe("fastcoref")
    return nlp, resolver

def _is_pronoun_text(txt: str) -> bool:
    return txt.strip().lower() in _PRON_SET

def _iter_cluster_items(cluster):
    """
    Yields (text, start_char, end_char, has_pos, pos_str)
    Handles spaCy Span *or* (start,end) tuples.
    """
    for item in cluster:
        if hasattr(item, "text") and hasattr(item, "start_char"):
            # spaCy Span
            pos = getattr(getattr(item, "root", None), "pos_", None)
            yield item.text, int(item.start_char), int(item.end_char), True, pos
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            # (start, end)
            s, e = int(item[0]), int(item[1])
            yield None, s, e, False, None
        # else: ignore

def _choose_representative(mentions: List[Dict[str, Any]]) -> str:
    non_pr = [m for m in mentions if not m.get("is_pronoun")]
    src = non_pr if non_pr else mentions
    return max(src, key=lambda m: len(m["text"]))["text"] if src else ""

def _relation_heuristic(m1: str, m2: str) -> str:
    a, b = m1.lower(), m2.lower()
    if a == b:
        return "MATCH"
    if a in b or b in a:
        return "CONTAINS"
    if _is_pronoun_text(a) or _is_pronoun_text(b):
        return "ENT_PRON"
    return "OTHER"

def _add_sentence_ids_to_mentions(mentions: List[Dict[str, Any]],
                                  sentence_analyses: Optional[List[Dict[str, Any]]] = None):
    if not sentence_analyses:
        return
    # sentences carry absolute doc offsets in your pipeline: 'doc_start','doc_end'
    for m in mentions:
        s0, s1 = m.get("start_char"), m.get("end_char")
        sid = None
        for s in sentence_analyses:
            a, b = s.get("doc_start"), s.get("doc_end")
            if a is None or b is None:  # guard
                continue
            if s0 is not None and s1 is not None and a <= s0 and s1 <= b:
                sid = s.get("sentence_id")
                break
        if sid is not None:
            m["sentence_id"] = sid

def run_lingmess_coref(full_text: str,
                       nlp,
                       resolver=None,
                       sentence_analyses: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Returns {"chains":[{chain}], "resolved_text": str|None}
      chain := {
        "chain_id": int,
        "representative": str,
        "mentions":[{"text","start_char","end_char","is_pronoun","head_pos","sentence_id?"}],
        "edges":[{"i": int, "j": int, "relation": str}]
      }
    """
    doc = nlp(full_text)
    chains: List[Dict[str, Any]] = []
    if not hasattr(doc._, "coref_clusters"):
        return {"chains": chains, "resolved_text": None}

    for chain_id, cluster in enumerate(doc._.coref_clusters):
        mentions: List[Dict[str, Any]] = []
        for (txt, s, e, has_pos, pos) in _iter_cluster_items(cluster):
            mtxt = txt if txt is not None else full_text[s:e]
            if not mtxt:
                continue
            mentions.append({
                "text": mtxt,
                "start_char": s,
                "end_char": e,
                "is_pronoun": _is_pronoun_text(mtxt) if not has_pos else (pos == "PRON"),
                "head_pos": pos if has_pos else None,
            })

        if not mentions:
            continue

        # sentence ids
        _add_sentence_ids_to_mentions(mentions, sentence_analyses)

        # representative
        representative = _choose_representative(mentions)

        # rich "edges" (pairwise relationships). Keep it simple; UI can group by relation later.
        edges: List[Dict[str, Any]] = []
        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                rel = _relation_heuristic(mentions[i]["text"], mentions[j]["text"])
                edges.append({"i": i, "j": j, "relation": rel})

        chains.append({
            "chain_id": int(chain_id),
            "representative": representative,
            "mentions": mentions,
            "edges": edges,
        })

    # Resolved text (if fastcoref provides it)
    resolved_text = getattr(getattr(doc._, "resolved_text", None), "__str__", lambda: None)()
    if resolved_text is None and hasattr(doc._, "resolved_text"):
        # some builds expose it as a plain string
        rt = doc._.resolved_text
        if isinstance(rt, str):
            resolved_text = rt

    return {"chains": chains, "resolved_text": resolved_text}

"""
Created on Wed Sep 17 11:00:26 2025

@author: niran
"""

