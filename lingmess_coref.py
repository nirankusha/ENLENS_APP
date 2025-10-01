# -*- coding: utf-8 -*-
# lingmess_coref.py
# Utilities to run LingMess (fastcoref) and emit chains + rich "edges" for UI.

from __future__ import annotations
import os, re
from typing import List, Dict, Any, Optional, Tuple
import spacy                           # noqa
from fastcoref.spacy_component import spacy_component  # noqa (registers "fastcoref")

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

LINGMESS_CATEGORY_BY_NAME = {name: idx for idx, name in LINGMESS_CATEGORIES.items()}

_PRON_SET = {"he","she","it","they","him","her","them","his","hers","its","their","theirs"}

def ensure_fastcoref_component():
    """Ensure the LingMess fastcoref spaCy component is registered."""
    from fastcoref.spacy_component import spacy_component  # noqa: F401 (registration side-effect)
    return spacy_component

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

def _call_resolver_pair_scores(resolver, doc):
    """Best-effort helper to obtain LingMess pair scores from the resolver."""
    if resolver is None:
        return None

    for attr in ("get_pairwise_scores", "pair_scores"):
        fn = getattr(resolver, attr, None)
        if not callable(fn):
            continue
        try:
            return fn(doc)
        except TypeError:
            # Some implementations expose a cached property without doc argument.
            try:
                return fn()
            except Exception:
                continue
        except Exception:
            continue
    return None


def _coerce_index(value, span_to_idx):
    """Translate various representations of a mention into its index."""
    if value is None:
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        try:
            return int(value)
        except Exception:
            return None

    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None
        try:
            return int(txt)
        except ValueError:
            return None

    if isinstance(value, dict):
        for key in ("i", "j", "idx", "index", "mention", "mention_idx"):
            if key in value:
                idx = _coerce_index(value.get(key), span_to_idx)
                if idx is not None:
                    return idx
        start = value.get("start_char", value.get("start"))
        end = value.get("end_char", value.get("end"))
        if start is not None and end is not None:
            return _coerce_index((start, end), span_to_idx)
        span = value.get("span") or value.get("pair")
        if span is not None:
            return _coerce_index(span, span_to_idx)
        indices = value.get("indices")
        if indices is not None:
            return _coerce_index(indices, span_to_idx)
        return None

    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            key = (int(value[0]), int(value[1]))
        except Exception:
            return None
        return span_to_idx.get(key)

    return None


def _normalize_category_label(value):
    """Return (id, label) for a LingMess relation descriptor."""
    if value is None:
        return None, None

    if isinstance(value, dict):
        for key in ("category", "category_id", "relation", "relation_id", "label", "id", "lingmess_id"):
            if key in value:
                cid, lbl = _normalize_category_label(value.get(key))
                if cid is not None or lbl is not None:
                    return cid, lbl
        return None, None

    if isinstance(value, (list, tuple)):
        for item in value:
            cid, lbl = _normalize_category_label(item)
            if cid is not None or lbl is not None:
                return cid, lbl
        return None, None

    if isinstance(value, str):
        txt = value.strip().upper().replace("-", "_")
        if not txt:
            return None, None
        if txt in _LINGMESS_CATEGORY_BY_NAME:
            cat_id = _LINGMESS_CATEGORY_BY_NAME[txt]
            return cat_id, LINGMESS_CATEGORIES.get(cat_id)
        try:
            return _normalize_category_label(int(txt))
        except ValueError:
            return None, txt

    if isinstance(value, (int, float)):
        try:
            cat_id = int(value)
        except Exception:
            return None, None
        return cat_id, LINGMESS_CATEGORIES.get(cat_id)

    return None, None


def _register_relation(result, idx_a, idx_b, cat_id, cat_label, score):
    if idx_a is None or idx_b is None or idx_a == idx_b:
        return
    key = tuple(sorted((int(idx_a), int(idx_b))))
    if key not in result:
        result[key] = {"id": cat_id, "label": cat_label, "score": score}
        return
    current = result[key]
    # Prefer entries with an explicit score or label.
    if current.get("score") is None and score is not None:
        current["score"] = score
    if current.get("id") is None and cat_id is not None:
        current["id"] = cat_id
    if current.get("label") is None and cat_label is not None:
        current["label"] = cat_label


def _parse_pair_entries(node, span_to_idx):
    """Flatten various LingMess pair-score structures to {(i,j): {...}}."""
    result: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if node is None:
        return result

    if isinstance(node, dict):
        idx_keys_a = ("i", "i_idx", "idx1", "index1", "antecedent", "antecedent_idx", "span1", "mention1", "first", "source")
        idx_keys_b = ("j", "j_idx", "idx2", "index2", "anaphor", "anaphor_idx", "span2", "mention2", "second", "target")

        idx_a = None
        idx_b = None
        for key in idx_keys_a:
            if key in node:
                idx_a = _coerce_index(node.get(key), span_to_idx)
                if idx_a is not None:
                    break
        for key in idx_keys_b:
            if key in node:
                idx_b = _coerce_index(node.get(key), span_to_idx)
                if idx_b is not None:
                    break

        if idx_a is None or idx_b is None:
            pair_val = node.get("pair") or node.get("pairs") or node.get("indices")
            if isinstance(pair_val, (list, tuple)) and len(pair_val) >= 2:
                idx_a = _coerce_index(pair_val[0], span_to_idx) if idx_a is None else idx_a
                idx_b = _coerce_index(pair_val[1], span_to_idx) if idx_b is None else idx_b

        score = None
        for key in ("score", "prob", "probability", "confidence", "logit", "value"):
            if key in node:
                try:
                    score = float(node.get(key))
                except Exception:
                    score = None
                break

        cat_id, cat_label = _normalize_category_label(node)
        if (idx_a is not None and idx_b is not None) and (cat_id is not None or cat_label is not None):
            _register_relation(result, idx_a, idx_b, cat_id, cat_label, score)
        else:
            for value in node.values():
                nested = _parse_pair_entries(value, span_to_idx)
                for key, data in nested.items():
                    existing = result.get(key)
                    if existing is None:
                        result[key] = data
                    else:
                        _register_relation(result, key[0], key[1], data.get("id"), data.get("label"), data.get("score"))
        return result

    if isinstance(node, (list, tuple, set)):
        for item in node:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                idx_a = _coerce_index(item[0], span_to_idx)
                idx_b = _coerce_index(item[1], span_to_idx)
                score = None
                if len(item) > 3:
                    try:
                        score = float(item[3])
                    except Exception:
                        score = None
                cat_id, cat_label = _normalize_category_label(item[2])
                if idx_a is not None and idx_b is not None and (cat_id is not None or cat_label is not None):
                    _register_relation(result, idx_a, idx_b, cat_id, cat_label, score)
                    continue
            nested = _parse_pair_entries(item, span_to_idx)
            for key, data in nested.items():
                existing = result.get(key)
                if existing is None:
                    result[key] = data
                else:
                    _register_relation(result, key[0], key[1], data.get("id"), data.get("label"), data.get("score"))
        return result

    return result


def _cluster_pair_relations(pair_payload, cluster_idx, mentions):
    """Extract pair relations for a specific cluster."""
    if pair_payload is None:
        return {}

    span_to_idx = {}
    for idx, mention in enumerate(mentions):
        s = mention.get("start_char")
        e = mention.get("end_char")
        if s is None or e is None:
            continue
        span_to_idx.setdefault((int(s), int(e)), idx)

    candidates = []
    if isinstance(pair_payload, dict):
        for key in (cluster_idx, str(cluster_idx)):
            if key in pair_payload:
                candidates.append(pair_payload[key])
        for key in ("clusters", "cluster_pairs", "pair_scores", "pairs", "scores"):
            val = pair_payload.get(key)
            if isinstance(val, (list, tuple)) and cluster_idx < len(val):
                candidates.append(val[cluster_idx])
    elif isinstance(pair_payload, (list, tuple)):
        if cluster_idx < len(pair_payload):
            candidates.append(pair_payload[cluster_idx])
        # Some implementations return a flat list of entries.
        candidates.extend(pair_payload)
    else:
        candidates.append(pair_payload)

    relations: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for candidate in candidates:
        parsed = _parse_pair_entries(candidate, span_to_idx)
        for key, data in parsed.items():
            existing = relations.get(key)
            if existing is None:
                relations[key] = data
            else:
                _register_relation(relations, key[0], key[1], data.get("id"), data.get("label"), data.get("score"))

    return relations

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
    
    pair_payload = _call_resolver_pair_scores(resolver, doc)
    
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
        cluster_pairs = _cluster_pair_relations(pair_payload, int(chain_id), mentions)
        edges: List[Dict[str, Any]] = []
        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                heuristic_rel = _relation_heuristic(mentions[i]["text"], mentions[j]["text"])
                key = (i, j) if (i, j) in cluster_pairs else (j, i) if (j, i) in cluster_pairs else (min(i, j), max(i, j))
                pair_info = cluster_pairs.get(key)

                lingmess_id = None
                lingmess_relation = None
                lingmess_score = None
                if pair_info:
                    lingmess_id = pair_info.get("id")
                    lingmess_relation = pair_info.get("label")
                    if isinstance(lingmess_relation, str):
                        norm_rel = lingmess_relation.strip().upper().replace("-", "_")
                        lingmess_relation = norm_rel or None
                    lingmess_score = pair_info.get("score")

                relation = None
                relation_source = "heuristic"
                if lingmess_relation and lingmess_relation not in {"NO_RELATION", "NOT_COREF"}:
                    relation = lingmess_relation
                    relation_source = "lingmess"
                elif lingmess_id is not None and lingmess_id not in (-1, 0):
                    relation = LINGMESS_CATEGORIES.get(lingmess_id, heuristic_rel)
                    relation_source = "lingmess"

                if relation is None:
                    relation = heuristic_rel or "OTHER"

                edge = {
                    "i": i,
                    "j": j,
                    "relation": relation,
                    "heuristic": heuristic_rel,
                    "relation_source": relation_source,
                }
                if lingmess_id is not None:
                    edge["lingmess_id"] = lingmess_id
                if lingmess_relation is not None:
                    edge["lingmess_relation"] = lingmess_relation
                if lingmess_score is not None:
                    edge["lingmess_score"] = lingmess_score

                edges.append(edge)

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

