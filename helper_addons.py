# -*- coding: utf-8 -*-
"""
helper_addons.py
----------------
Add-on helpers that work WITH your existing helper.py and models.

This module imports your existing models/pipelines from `helper` and exposes
normalized utilities used by the three updated pipelines, without modifying
your original helper.py.

If you prefer a single file, copy the section:
    ### BEGIN: MERGE INTO helper.py
    ...
    ### END: MERGE INTO helper.py
into your existing helper.py.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

# Import your existing stuff (models, nlp, I/O, classifiers, etc.)
from helper import (
    nlp, device, SDG_TARGETS,
    extract_text_from_pdf_robust, preprocess_pdf_text, extract_and_filter_sentences,
    classify_sentence_bert, classify_sentence_similarity, determine_dual_consensus,
    analyze_full_text_coreferences,
    expand_to_full_phrase, normalize_span_for_chaining,
)

# Try to import the tokenizer/model you already use
try:
    from helper import bert_tokenizer, bert_model  # your existing BERT classifier assets
except Exception:
    bert_tokenizer = None
    bert_model = None

# =============================================================================
# ### BEGIN: MERGE INTO helper.py  (copy into your helper.py if you want one file)
# =============================================================================

# ---- Lightweight Interval / IntervalTree ------------------------------------------------
class Interval:
    __slots__ = ("begin", "end", "data")
    def __init__(self, begin: int, end: int, data: Any=None):
        if end < begin:
            raise ValueError("Interval end must be >= begin")
        self.begin = int(begin)
        self.end   = int(end)
        self.data  = data
    def __repr__(self):
        return f"Interval({self.begin},{self.end},{self.data!r})"

class IntervalTree:
    def __init__(self, intervals: Optional[List["Interval"]] = None):
        self._ivals: List[Interval] = list(intervals or [])
        self._ivals.sort(key=lambda x: x.begin)
    def add(self, interval: "Interval"):
        self._ivals.append(interval)
        self._ivals.sort(key=lambda x: x.begin)
    def at(self, point: int) -> List[Any]:
        p = int(point)
        out: List[Any] = []
        for iv in self._ivals:
            if iv.begin <= p < iv.end:
                out.append(iv.data)
            if iv.begin > p:
                break
        return out
    def search(self, begin: int, end: int) -> List[Any]:
        b, e = int(begin), int(end)
        out: List[Any] = []
        for iv in self._ivals:
            if iv.end <= b:
                continue
            if iv.begin >= e:
                break
            if (iv.begin < e) and (iv.end > b):
                out.append(iv.data)
        return out

# ---- Sentence index over spaCy sentences (dict payloads) --------------------------------
def build_sentence_index(full_text: str) -> Tuple[Dict[int, Tuple[int,int]], IntervalTree]:
    """
    Returns:
      sid2span: {sid -> (start_char, end_char)}
      sent_tree: IntervalTree with payload dicts {"sid","start","end"}
    Aligned to spaCy doc.sents if available; falls back to regex segmentation.
    """
    try:
        doc = nlp(full_text)
        sid2span = {i: (s.start_char, s.end_char) for i, s in enumerate(doc.sents)}
        tree = IntervalTree([Interval(st, en, {"sid": i, "start": st, "end": en}) for i,(st,en) in sid2span.items()])
        return sid2span, tree
    except Exception:
        import re
        sents = [s for s in re.split(r"(?<=[.!?])\s+", full_text) if s]
        sid2span: Dict[int, Tuple[int,int]] = {}
        pos = 0
        for i, s in enumerate(sents):
            idx = full_text.find(s, pos)
            if idx < 0: idx = full_text.find(s)
            sid2span[i] = (idx, idx + len(s))
            pos = idx + len(s)
        tree = IntervalTree([Interval(st, en, {"sid": i, "start": st, "end": en}) for i,(st,en) in sid2span.items()])
        return sid2span, tree

# ---- Normalize chain mentions (tuple or dict -> dict) -----------------------------------
def normalize_chain_mentions(chains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm: List[Dict[str, Any]] = []
    for i, ch in enumerate(chains or []):
        cid = ch.get("chain_id", i)
        rep = ch.get("representative", "")
        ments = []
        for m in ch.get("mentions", []) or []:
            if isinstance(m, dict):
                s = int(m.get("start_char", -1)); e = int(m.get("end_char", -1)); t = str(m.get("text", ""))
            else:
                try:
                    s, e, t = int(m[0]), int(m[1]), str(m[2])
                except Exception:
                    continue
            if 0 <= s < e:
                ments.append({"start_char": s, "end_char": e, "text": t})
        norm.append({"chain_id": int(cid), "representative": rep, "mentions": ments})
    return norm

# ---- Offset-aware token attribution using your existing model ---------------------------
def token_importance_ig(text: str, class_id: int):
    """
    Returns (tokens, scores, offsets) using embedding-level IG.
    - tokens: list[str]
    - scores: list[float]  (per-token saliency)
    - offsets: list[tuple[int,int]]  (char spans in the input text)
    Falls back to a whitespace heuristic if tokenizer/model are not available.
    """
    # --- fast fallback if model not wired ---
    if bert_tokenizer is None or bert_model is None:
        import re
        tokens = re.findall(r"\S+", text)
        pos, offs = 0, []
        for t in tokens:
            i = text.find(t, pos)
            offs.append((i, i+len(t)))
            pos = i + len(t)
        scores = [float(len(t)) for t in tokens]  # length proxy
        return tokens, scores, offs

    import torch
    bert_model.eval()

    # Encode with offsets if supported by a fast tokenizer
    enc = bert_tokenizer(
        text, return_offsets_mapping=True, return_tensors="pt",
        truncation=True, max_length=512
    )
    offsets = enc.pop("offset_mapping", None)  # shape: (1, seq, 2) for fast tokenizers
    input_ids = enc["input_ids"]
    attn_mask = enc.get("attention_mask", None)

    # Move to the model's device/dtype
    model_device = next(bert_model.parameters()).device
    input_ids = input_ids.to(model_device)
    if attn_mask is not None:
        attn_mask = attn_mask.to(model_device)

    # Obtain the embedding module generically
    try:
        emb_layer = bert_model.get_input_embeddings()
    except Exception:
        # Fallback for unusual wrappers
        emb_layer = (
            getattr(getattr(bert_model, "bert", None), "embeddings", None) or
            getattr(getattr(bert_model, "roberta", None), "embeddings", None) or
            getattr(getattr(bert_model, "distilbert", None), "embeddings", None)
        )
        if emb_layer is None or not hasattr(emb_layer, "word_embeddings"):
            # Cannot access embeddings -> fallback
            tokens = bert_tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            scores = [0.0] * len(tokens)
            if offsets is not None:
                offs = [(int(a), int(b)) for a, b in offsets[0].tolist()]
            else:
                # whitespace fallback
                import re
                toks = re.findall(r"\S+", text)
                pos, offs = 0, []
                for t in toks:
                    i = text.find(t, pos); offs.append((i, i+len(t))); pos = i + len(t)
            return tokens, scores, offs

        emb_layer = emb_layer.word_embeddings  # nn.Embedding

    with torch.enable_grad():
        # Get embeddings for input ids
        inputs_embeds = emb_layer(input_ids)  # (1, seq, hidden)
        inputs_embeds = inputs_embeds.detach().requires_grad_(True)

        # Simple embedding-level Integrated Gradients
        steps = int(os.environ.get("IG_STEPS", "8"))
        steps = max(1, min(steps, 32))
        baseline = torch.zeros_like(inputs_embeds)

        total_grads = torch.zeros_like(inputs_embeds)
        for k in range(1, steps + 1):
            alpha = float(k) / steps
            x = baseline + alpha * (inputs_embeds - baseline)
            x.retain_grad()
            bert_model.zero_grad(set_to_none=True)
            out = bert_model(inputs_embeds=x, attention_mask=attn_mask).logits  # (1, num_labels)
            tgt = out[0, int(class_id) % out.shape[-1]]
            tgt.backward(retain_graph=True)
            if x.grad is not None:
                total_grads = total_grads + x.grad.detach()

        avg_grads = total_grads / steps
        attributions = (inputs_embeds - baseline) * avg_grads  # (1, seq, hidden)
        sal = attributions.norm(dim=-1).squeeze(0)            # (seq,)

    # Convert to Python lists
    tokens = bert_tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
    scores = sal.detach().cpu().tolist()

    # Offsets
    if offsets is not None:
        offs = [(int(a), int(b)) for a, b in offsets[0].tolist()]
    else:
        # Fallback offsets if tokenizer isn’t "fast"
        # Try greedy substring mapping; if it fails, use whitespace chunks
        try:
            decoded = bert_tokenizer.convert_ids_to_tokens(input_ids[0].tolist(), skip_special_tokens=False)
            # Basic heuristic: ignore special tokens, derive spans by iterating text
            import re
            pos, offs = 0, []
            for tok in decoded:
                clean = tok.replace("Ġ", "").replace("##", "").replace("▁", "")
                if clean == "" or tok in ("[CLS]", "[SEP]", "[PAD]"):
                    offs.append((pos, pos))
                    continue
                i = text.find(clean, pos)
                if i < 0:
                    offs.append((pos, pos))
                else:
                    offs.append((i, i + len(clean)))
                    pos = i + len(clean)
        except Exception:
            import re
            toks = re.findall(r"\S+", text)
            pos, offs = 0, []
            for t in toks:
                i = text.find(t, pos); offs.append((i, i+len(t))); pos = i + len(t)

    return tokens, scores, offs

# ---- Masking-based span salience (sentence-local) --------------------------------------
def compute_span_importances(text: str, target_class: int, max_span_len: int = 4) -> List[Dict[str, Any]]:
    """
    Returns a list of {start, end, score, text} (sentence-local offsets).
    If you already have a project masking scorer, replace this body to call it.
    """
    import re
    spans: List[Dict[str, Any]] = []
    tokens = list(re.finditer(r"\S+", text))
    n = len(tokens)
    for i in range(n):
        for L in range(1, max_span_len+1):
            j = i + L
            if j > n: break
            s = tokens[i].start(); e = tokens[j-1].end()
            spans.append({"start": s, "end": e, "score": float(e-s), "text": text[s:e]})
    return spans

# ---- Minimal clusters block for unified outputs ----------------------------------------
def prepare_clusters(production_output: Dict[str, Any]) -> Dict[str, Any]:
    chains = (production_output.get("coreference_analysis") or {}).get("chains", [])
    sents = production_output.get("sentence_analyses", [])
    sent_chains: Dict[int, List[int]] = {}
    for sa in sents:
        sid = int(sa.get("sentence_id", 0))
        present: List[int] = []
        for sp in (sa.get("span_analysis") or []):
            ci = ((sp.get("coreference_analysis") or {}).get("chain_id"))
            if isinstance(ci, int): present.append(ci)
        sent_chains[sid] = sorted(set(present))
    clusters: List[List[int]] = []
    seen = set()
    for sid, cids in sent_chains.items():
        if sid in seen: continue
        group = [sid]; seen.add(sid)
        for sid2, cids2 in sent_chains.items():
            if sid2 in seen: continue
            if set(cids) & set(cids2):
                group.append(sid2); seen.add(sid2)
        clusters.append(sorted(group))
    graphs_json = {"nodes": [{"id": sid} for sid in sent_chains.keys()],
                   "edges": [{"source": a, "target": b}
                             for cl in clusters for i, a in enumerate(cl) for b in cl[i+1:]]}
    return {"clusters": clusters, "clusters_dict": {str(i): cl for i, cl in enumerate(clusters)}, "graphs_json": graphs_json}

# =============================================================================
# ### END: MERGE INTO helper.py
# =============================================================================

"""
Created on Sat Aug 16 17:04:29 2025

@author: niran
"""

