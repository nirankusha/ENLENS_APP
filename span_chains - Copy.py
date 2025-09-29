# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, Literal, Iterable, Callable 
import json

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

from helper_addons import (
    Interval, IntervalTree, build_sentence_index, normalize_chain_mentions,
    token_importance_ig, compute_span_importances, prepare_clusters, build_ngram_trie, shortlist_by_trie,
    build_cooc_graph, shortlist_by_cooc, fuse_shortlists, build_spacy_trie_with_idf, build_ngram_index
)

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

def _map_span_to_chain(
    abs_s: int,
    abs_e: int,
    chain_trees: Dict[int, "IntervalTree"],
    chains: List[Dict[str, Any]],
    *,
    cand_chain_ids: Optional[Iterable[int]] = None,   # NEW: shortlist
    scorer: Optional[Callable[[int, int, Dict[str, Any]], float]] = None,  # NEW: optional pair scorer
    threshold: Optional[float] = None,                # NEW: scorer threshold
) -> Dict[str, Any]:
    """
    Map [abs_s, abs_e) to a coref chain.

    Backward-compatible:
      - If no cand_chain_ids and no scorer: identical behavior to the original.
      - Preserves 'related_mentions' construction & probe logic.

    Extensions:
      - If cand_chain_ids is provided, restricts consideration to those chains.
      - If scorer is provided, selects the best candidate >= threshold; falls back to probe.
    """
    probes = [abs_s, (abs_s + abs_e) // 2, max(abs_s, abs_e - 1)]

    # Prepare chain iterator (respecting shortlist if present)
    if cand_chain_ids is not None:
        cand_set = set(int(x) for x in cand_chain_ids)
        chains_iter = [ch for ch in chains if int(ch.get("chain_id", -1)) in cand_set]
    else:
        chains_iter = chains

    # --- Optional scorer path (keeps your logging; only changes selection policy) ---
    if scorer is not None and chains_iter:
        scored: List[tuple[int, float]] = []
        for ch in chains_iter:
            try:
                s = float(scorer(abs_s, abs_e, ch))
            except Exception:
                s = 0.0
            scored.append((int(ch.get("chain_id", -1)), s))
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and (threshold is None or scored[0][1] >= threshold):
            best_id = scored[0][0]
            # Build the SAME payload you currently return (incl. related mentions)
            for ch in chains_iter:
                if int(ch.get("chain_id", -1)) == best_id:
                    related = []
                    for m in ch.get("mentions", [])[:5]:
                        m0 = int(m.get("start_char", -1))
                        m1 = int(m.get("end_char", -1))
                        if not (m0 == abs_s and m1 == abs_e):
                            related.append({"text": m.get("text", ""), "coords": [m0, m1]})
                    return {
                        "chain_found": True,
                        "chain_id": best_id,
                        "representative": ch.get("representative", ""),
                        "related_mentions": related,
                        # optional diagnostic; harmless if ignored by callers
                        "decision": "scorer"
                    }
        # If scorer fails to pick, we fall through to probe logic (below)

    # --- Original PROBE logic (preserved) over the (possibly restricted) chains ---
    for ch in chains_iter:
        tree = chain_trees.get(int(ch.get("chain_id", -1)))
        if not tree:
            continue
        if any(tree.at(p) for p in probes):
            related = []
            for m in ch.get("mentions", [])[:5]:
                m0 = int(m.get("start_char", -1))
                m1 = int(m.get("end_char", -1))
                if not (m0 == abs_s and m1 == abs_e):
                    related.append({"text": m.get("text", ""), "coords": [m0, m1]})
            return {
                "chain_found": True,
                "chain_id": int(ch.get("chain_id", -1)),
                "representative": ch.get("representative", ""),
                "related_mentions": related,
                "decision": "probe"  # optional diagnostic tag
            }

    # No match
    return {"chain_found": False, "decision": "none"}  # decision is optional
"""
Created on Fri Sep 12 14:08:39 2025

@author: niran
"""

