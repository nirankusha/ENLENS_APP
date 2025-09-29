# -*- coding: utf-8 -*-
"""
helper_addons.py
----------------
Add-on helpers that work WITH existing helper.py and models.

This module imports your existing models/pipelines from `helper` and exposes
normalized utilities used by the three updated pipelines, without modifying
your original helper.py.
"""
from __future__ import annotations
from collections import defaultdict, Counter
import math
import re
from typing import Dict, List, Tuple, Iterable, Optional, Any, Set
import os 
from dataclasses import dataclass, field



# Import your existing stuff (models, nlp, I/O, classifiers, etc.)
from helper import (
    nlp, device, SDG_TARGETS,
    extract_text_from_pdf_robust, preprocess_pdf_text, extract_and_filter_sentences,
    classify_sentence_bert, classify_sentence_similarity, determine_dual_consensus,
    analyze_full_text_coreferences,
    expand_to_full_phrase, normalize_span_for_chaining
)

# Try to import the tokenizer/model you already use
try:
    from helper import bert_tokenizer, bert_model  # your existing BERT classifier assets
except Exception:
    bert_tokenizer = None
    bert_model = None

TOKEN_JOIN = "§"


# =============================================================================
# ### Add to helper = new data classes and preprocessing methods
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

def build_alignment_map(original: str, resolved: str, replacements: List[Tuple[int,int,str]]) -> List[Tuple[int,int,int,int]]:
    """
    Build sparse alignment between original and resolved after replacements.
    replacements: list of (start,end,rep_text) applied left→right on original.
    Returns list of (orig_start, orig_end, res_start, res_end) ranges.
    """
    spans = []
    o_cursor = 0
    r_cursor = 0
    last = 0
    res_text = []
    for s,e,rep in replacements:
        # keep original chunk
        if s > last:
            chunk = original[last:s]
            res_text.append(chunk)
            L = len(chunk)
            spans.append((last, s, r_cursor, r_cursor+L))
            r_cursor += L
        # replaced chunk
        rep_str = str(rep)
        res_text.append(rep_str)
        spans.append((s, e, r_cursor, r_cursor+len(rep_str)))
        r_cursor += len(rep_str)
        last = e
    # tail
    if last < len(original):
        chunk = original[last:]
        res_text.append(chunk)
        L = len(chunk)
        spans.append((last, len(original), r_cursor, r_cursor+L))
        r_cursor += L
    # sanity: "".join(res_text) == resolved
    return spans

def debug_coref_clusters(production_output):
    chains = (production_output.get("coreference_analysis") or {}).get("chains") or []
    stats = []
    for ch in chains:
        cid = ch["chain_id"]
        m = ch.get("mentions", [])
        e = ch.get("edges", [])
        tags = {}
        for ed in e:
            tags[ed["tag"]] = tags.get(ed["tag"], 0) + 1
        stats.append({
            "chain": cid,
            "mentions": len(m),
            "sent_spread": len({mm.get("sent_id") for mm in m if mm.get("sent_id") is not None}),
            "tags": tags,
            "repr": ch.get("representative","")[:40]
        })
    return stats

# --- BASIC TEXT NORMALIZATION / TOKEN HELPERS (new) ---
_STOP = set(getattr(nlp.Defaults, "stop_words", set()))
_WORD_RE = re.compile(r"\w+", re.U)

def _norm_tokens(text: str) -> List[str]:
    doc = nlp(text)
    out = []
    for t in doc:
        if t.is_space: 
            continue
        if t.is_punct:
            continue
        form = t.lemma_.lower() if t.lemma_ else t.text.lower()
        form = form.strip()
        if not form or form in _STOP:
            continue
        out.append(form)
    return out

def _head_lemmas(text: str) -> List[str]:
    doc = nlp(text)
    heads = []
    for nc in doc.noun_chunks:
        heads.append(nc.root.lemma_.lower() if nc.root.lemma_ else nc.root.text.lower())
    return heads

def _char_ngrams(s: str, n: int = 4) -> List[str]:
    s = re.sub(r"\s+", " ", s.lower()).strip()
    s = re.sub(r"[^\w ]+", "", s)
    if len(s) < n: 
        return []
    return [s[i:i+n] for i in range(len(s)-n+1)]

def _token_ngrams(tokens: List[str], n: int) -> List[str]:
    return [TOKEN_JOIN.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# -------------------------------
# TRIE (posting lists with counts)
# -------------------------------
class _TrieNode(dict):
    __slots__ = ("post",)
    def __init__(self):
        super().__init__()
        self.post: Counter = Counter()  # chain_id -> count

def build_ngram_trie(chains, *, char_n: int = 4, token_ns: Tuple[int, ...] = (2, 3)):
    idx = build_ngram_index(chains, token_ns=token_ns, char_n=char_n, build_trie=True)
    return idx.ensure_trie(), idx.idf, idx.chain_grams

def build_spacy_trie_with_idf(chains, *, token_ns: Tuple[int, ...] = (2,3,4), char_n: Optional[int] = None):
    idx = build_ngram_index(chains, token_ns=token_ns, char_n=char_n, build_trie=True)
    return idx.ensure_trie(), idx.idf, idx.chain_grams

def _trie_lookup(root: _TrieNode, gram: str) -> Counter:
    node = root
    for ch in gram:
        node = node.get(ch)
        if node is None:
            return Counter()
    return node.post

def shortlist_by_trie(
    span_text: str,
    *,
    root: _TrieNode,
    idf: Dict[str, float],
    chain_grams: Dict[int, Counter],
    char_n: Optional[int] = None,
    token_ns: Tuple[int, ...] = (2, 3, 4),
    topk: int = 5,
    tau: float = 0.18,
) -> List[Tuple[int, float]]:
    """Return [(chain_id, score)] by IDF-weighted Jaccard over grams."""
    toks = _norm_tokens(span_text)
    heads = _head_lemmas(span_text)
    Gs: Set[str] = set()
    if char_n is not None:
        Gs |= set(_char_ngrams(span_text, char_n))
    for n in token_ns:
        Gs |= set(_token_ngrams(toks, n))
    Gs |= set(heads)

    if not Gs:
        return []

    num = Counter()
    den = Counter()
    # precompute union denominators per chain: |Gs ∪ Gc|_idf
    span_idf_sum = sum(idf.get(g, 0.0) for g in Gs)
    for g in Gs:
        post = _trie_lookup(root, g)
        w = idf.get(g, 0.0)
        for cid in post:
            num[cid] += w

    # denominator: sum idf over union
    out = []
    for cid, inter_w in num.items():
        cg = chain_grams.get(cid, Counter())
        chain_idf_sum = sum(idf.get(g, 0.0) for g in cg.keys())
        union_w = span_idf_sum + chain_idf_sum - inter_w
        score = (inter_w / union_w) if union_w > 0 else 0.0
        if score >= tau:
            out.append((cid, score))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:topk]

# ------------------------------------
# CO-OCCURRENCE GRAPH (PPMI row space)
# ------------------------------------

def build_cooc_graph(
    full_text: str,
    *,
    window: int = 5,
    min_count: int = 2,
    topk_neighbors: int = 10,
    mode: str = "hf",                   # "hf" | "spacy"
    hf_tokenizer=None,
    cooc_impl: str = "fast",            # "fast" | "py"
    cooc_max_tokens: int = 50000,
    cache_key: int | None = None,
) -> Tuple[Dict[str, int], List[Counter], Dict[int, float]]:
    """
    Build a PPMI-based co-occurrence graph.
    Returns:
      vocab: mapping (string ids -> row idx) for hf; tokens -> row idx for spacy
      rows:  List[Counter]: row -> {col: ppmi}
      row_norms: {row: l2_norm}
    Notes:
      - "fast" uses NumPy/SciPy and vectorized window shifts (much faster).
      - "py" is a guarded legacy implementation (pure Python).
      - Toggle debug logs with: os.environ["COOC_DEBUG"]="1"
    """
    import os, math
    DEBUG = os.environ.get("COOC_DEBUG", "0") not in ("0", "", "false", "False", "FALSE")
    def dbg(msg: str):
        if DEBUG:
            print(f"[cooc] {msg}")

    # -------------------- Simple in-process cache --------------------
    global _COOC_CACHE
    try:
        _ = _COOC_CACHE
    except NameError:
        _COOC_CACHE = {}
    ck = None
    if cache_key is not None:
        ck = (cache_key, window, min_count, topk_neighbors, mode, cooc_impl)
        hit = _COOC_CACHE.get(ck)
        if hit:
            return hit

    # -------------------- Tokenize & map to contiguous row ids --------------------
    if mode == "hf":
        if hf_tokenizer is None:
            raise ValueError("hf_tokenizer must be provided when mode='hf'")
        ids = hf_tokenizer(full_text, add_special_tokens=False, return_attention_mask=False).input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        tokens = [int(t) for t in ids if isinstance(t, int)]
        if len(tokens) > cooc_max_tokens:
            tokens = tokens[:cooc_max_tokens]
        freq = Counter(tokens)
        valid = sorted([tid for tid, c in freq.items() if c >= min_count])
        if not valid:
            return {}, [], {}
        id2row = {tid: i for i, tid in enumerate(valid)}
        stream = [id2row[t] for t in tokens if t in id2row]
        V = len(id2row)
        if not stream or V == 0:
            return {}, [], {}
        make_vocab = lambda: {str(tid): rid for tid, rid in id2row.items()}
        dbg(f"mode=hf V={V} stream_len={len(stream)} min={min(stream)} max={max(stream)}")
    else:
        # expects module-level: nlp, _STOP
        doc = nlp(full_text)
        toks = [
            (t.lemma_ or t.text).lower()
            for t in doc
            if (not t.is_space and not t.is_punct and (t.lemma_ or t.text).lower() not in _STOP)
        ]
        if len(toks) > cooc_max_tokens:
            toks = toks[:cooc_max_tokens]
        freq = Counter(toks)
        valid = sorted([tok for tok, c in freq.items() if c >= min_count])
        if not valid:
            return {}, [], {}
        tok2row = {tok: i for i, tok in enumerate(valid)}
        stream = [tok2row[t] for t in toks if t in tok2row]
        V = len(tok2row)
        if not stream or V == 0:
            return {}, [], {}
        make_vocab = lambda: dict(tok2row)
        dbg(f"mode=spacy V={V} stream_len={len(stream)} min={min(stream)} max={max(stream)}")

    # Sanity + clamp
    if max(stream) >= V or min(stream) < 0:
        stream = [x for x in stream if 0 <= x < V]
        if not stream:
            return make_vocab(), [], {}

    # -------------------- FAST IMPLEMENTATION --------------------
    if cooc_impl == "fast":
        import numpy as np
        try:
            import scipy.sparse as sp
        except Exception:
            sp = None

        a = np.asarray(stream, dtype=np.int32)
        n = a.size
        W = max(1, int(window))
        if n == 0:
            return make_vocab(), [], {}

        # Build (row, col) pairs by shifting
        rows_idx = []
        cols_idx = []
        for k in range(1, W + 1):
            if n - k <= 0:
                break
            rows_idx.append(a[:-k]); cols_idx.append(a[k:])   # (i, i+k)
            rows_idx.append(a[k:]);  cols_idx.append(a[:-k])  # (i+k, i)
        if not rows_idx:
            return make_vocab(), [Counter() for _ in range(V)], {i: 0.0 for i in range(V)}

        R = np.concatenate(rows_idx)
        C = np.concatenate(cols_idx)

        if sp is not None:
            # aggregate via sparse COO->CSR
            M = sp.coo_matrix((np.ones_like(R, dtype=np.float32), (R, C)), shape=(V, V), dtype=np.float32).tocsr()
            row_sums = np.asarray(M.sum(axis=1)).ravel()
            col_sums = np.asarray(M.sum(axis=0)).ravel()
            total = float(row_sums.sum())
            if total <= 0:
                vocab = make_vocab()
                rows = [Counter() for _ in range(V)]
                row_norms = {i: 0.0 for i in range(V)}
                out = (vocab, rows, row_norms)
                if ck: _COOC_CACHE[ck] = out
                return out

            # PMI on nonzeros
            M = M.tocoo()
            cnt = M.data.astype(np.float64)
            pr = row_sums[M.row].astype(np.float64)
            pc = col_sums[M.col].astype(np.float64)
            pmi = np.log((cnt * total) / (pr * pc) + 1e-12)
            pmi[pmi < 0] = 0.0
            P = sp.coo_matrix((pmi.astype(np.float32), (M.row, M.col)), shape=(V, V)).tocsr()

            # top-k prune per row
            if topk_neighbors and topk_neighbors > 0:
                k = int(topk_neighbors)
                P = P.tolil()
                for r in range(V):
                    if P.rows[r] and len(P.rows[r]) > k:
                        data = np.asarray(P.data[r])
                        idx = np.argpartition(data, -k)[-k:]
                        keep = set(int(P.rows[r][i]) for i in idx)
                        P.rows[r] = [j for j in P.rows[r] if j in keep]
                        P.data[r] = [v for j, v in zip(P.rows[r], P.data[r]) if j in keep]
                P = P.tocsr()

            # Convert back to List[Counter] + norms
            rows: List[Counter] = []
            row_norms: Dict[int, float] = {}
            for r in range(V):
                s, e = P.indptr[r], P.indptr[r+1]
                cols = P.indices[s:e]
                data = P.data[s:e]
                rows.append(Counter({int(j): float(v) for j, v in zip(cols, data) if v > 0.0}))
                row_norms[r] = float(np.sqrt((data ** 2).sum())) if data.size else 0.0

            vocab = make_vocab()
            out = (vocab, rows, row_norms)
            if ck: _COOC_CACHE[ck] = out
            return out

        # NumPy-only aggregation fallback
        from collections import defaultdict
        agg = defaultdict(float)
        for r, c in zip(R.tolist(), C.tolist()):
            agg[(r, c)] += 1.0

        row_sums = [0.0] * V
        col_sums = [0.0] * V
        for (r, c), v in agg.items():
            row_sums[r] += v
            col_sums[c] += v
        total = float(sum(row_sums))
        rows: List[Counter] = [Counter() for _ in range(V)]
        if total > 0:
            for (r, c), v in agg.items():
                val = math.log((v * total) / (row_sums[r] * col_sums[c] + 1e-12) + 1e-12)
                if val > 0:
                    rows[r][c] = val
        # prune
        if topk_neighbors and topk_neighbors > 0:
            k = int(topk_neighbors)
            for r in range(V):
                if len(rows[r]) > k:
                    rows[r] = Counter(dict(sorted(rows[r].items(), key=lambda kv: kv[1], reverse=True)[:k]))
        row_norms = {i: math.sqrt(sum(v*v for v in rows[i].values())) if rows[i] else 0.0 for i in range(V)}
        vocab = make_vocab()
        out = (vocab, rows, row_norms)
        if ck: _COOC_CACHE[ck] = out
        return out

    # -------------------- LEGACY PYTHON IMPLEMENTATION --------------------
    # Guarded, clear, and reasonably efficient for smaller docs.
    def _build_cooc_graph_py() -> Tuple[Dict[str,int], List[Counter], Dict[int,float]]:
        V_local = V
        rows = [Counter() for _ in range(V_local)]
        W_local = max(1, int(window))
        n_stream = len(stream)
        for i, r in enumerate(stream):
            if not (0 <= r < V_local):
                continue
            start = i - W_local if i - W_local > 0 else 0
            end   = i + W_local + 1 if i + W_local + 1 < n_stream else n_stream
            row = rows[r]
            for j in range(start, end):
                if j == i:
                    continue
                c = stream[j]
                if not (0 <= c < V_local):
                    continue
                row[c] = row.get(c, 0) + 1

        # counts -> PPMI
        row_sums = [sum(r.values()) for r in rows]
        col_sums = [0.0] * V_local
        for rr, r in enumerate(rows):
            for cc, cnt in r.items():
                col_sums[cc] += cnt
        total = float(sum(row_sums))
        if total > 0:
            eps = 1e-12
            for rr, r in enumerate(rows):
                pr = row_sums[rr] / total if row_sums[rr] > 0 else eps
                for cc, cnt in list(r.items()):
                    pc = col_sums[cc] / total if col_sums[cc] > 0 else eps
                    pij = cnt / total
                    val = math.log((pij / (pr * pc)) + eps)
                    if val > 0.0:
                        r[cc] = val
                    else:
                        del r[cc]

        # top-k prune
        if topk_neighbors and topk_neighbors > 0:
            k = int(topk_neighbors)
            for rr in range(V_local):
                if len(rows[rr]) > k:
                    rows[rr] = Counter(dict(sorted(rows[rr].items(), key=lambda kv: kv[1], reverse=True)[:k]))

        row_norms = {i: math.sqrt(sum(v*v for v in rows[i].values())) if rows[i] else 0.0
                     for i in range(V_local)}
        vocab = make_vocab()
        return vocab, rows, row_norms

    vocab, rows, row_norms = _build_cooc_graph_py()
    out = (vocab, rows, row_norms)
    if ck: _COOC_CACHE[ck] = out
    return out



def _embed_tokens(tokens: Iterable[str], vocab: Dict[str, int], rows: List[Counter], row_norms: Dict[int, float]) -> List[float]:
    # simple sum of row vectors then L2 norm
    acc = Counter()
    for tok in tokens:
        i = vocab.get(tok)
        if i is None:
            continue
        for j, val in rows[i].items():
            acc[j] += val
    norm = math.sqrt(sum(v*v for v in acc.values())) or 1.0
    # return sparse-as-dense(ish) vector (indices implied by dict)
    # for cosine, we'll just compute dot with chain vec similarly represented
    return [norm, acc]  # pack (norm, sparse_map)

def _cosine_sparse(v_s, v_c) -> float:
    ns, ms = v_s  # (norm, map)
    nc, mc = v_c
    if ns == 0 or nc == 0:
        return 0.0
    # dot over smaller map
    if len(ms) > len(mc):
        ms, mc = mc, ms
    dot = sum(val * mc.get(idx, 0.0) for idx, val in ms.items())
    return dot / (ns * nc)

def shortlist_by_cooc(
    span_text: str,
    *,
    chains: List[Dict[str, Any]],
    vocab: Dict[str, int],
    rows: List[Counter],
    row_norms: Dict[int, float],
    topk: int = 5,
    tau: float = 0.18,
    head_boost: float = 1.5,
) -> List[Tuple[int, float]]:
    # embed span
    toks = _norm_tokens(span_text)
    v_s = _embed_tokens(toks, vocab, rows, row_norms)
    # precompute (or lazy cache) chain vectors
    results = []
    for ch in chains:
        cid = int(ch.get("chain_id"))
        # chain tokens: heads + content words
        ctoks = []
        for m in ch.get("mentions", []):
            txt = m.get("text", "") if isinstance(m, dict) else str(m)
            ctoks.extend(_norm_tokens(txt))
            # add heads boosted
            ctoks.extend(_head_lemmas(txt) * int(head_boost))
        v_c = _embed_tokens(ctoks, vocab, rows, row_norms)
        score = _cosine_sparse(v_s, v_c)
        if score >= tau:
            results.append((cid, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:topk]

def fuse_shortlists(
    A: List[Tuple[int, float]],
    B: List[Tuple[int, float]],
    *,
    wA: float = 0.6,
    wB: float = 0.4,
    topk: int = 5,
) -> List[int]:
    sc = defaultdict(float)
    for cid, s in A:
        sc[cid] = max(sc[cid], wA * s)
    for cid, s in B:
        sc[cid] += wB * s
    out = sorted(sc.items(), key=lambda x: x[1], reverse=True)[:topk]
    return [cid for cid, _ in out]

# spaCy-token trie with IDF

# === Unified N-gram index for in-memory & persisted use ===

@dataclass
class NGramIndex:
    token_ns: Tuple[int, ...] = (2, 3, 4)
    char_n: Optional[int] = None
    idf: Dict[str, float] = field(default_factory=dict)
    chain_grams: Dict[int, Counter] = field(default_factory=dict)     # per-chain grams (for Jaccard denom)
    postings: Dict[str, Counter] = field(default_factory=dict)        # gram -> Counter(chain_id -> count)
    root: Optional["_TrieNode"] = None                                # built on demand for legacy lookup

    def ensure_trie(self) -> "_TrieNode":
        """Lazy-build a character-walk trie from postings (drop-in for shortlist_by_trie)."""
        if self.root is not None:
            return self.root
        r = _TrieNode()
        # grams are strings; walking characters reproduces existing trie shape
        for g, cnts in self.postings.items():
            node = r
            for ch in g:
                node = node.setdefault(ch, _TrieNode())
            for cid, c in cnts.items():
                node.post[int(cid)] += int(c)
        self.root = r
        return r

    def clear_trie(self) -> None:
        """Free the in-memory trie (postings stay)."""
        self.root = None

    def has_trie(self) -> bool:
        return self.root is not None

    def shortlist(self, span_text: str, *, topk: int = 5, tau: float = 0.18) -> List[Tuple[int, float]]:
        """Use existing IDF-weighted Jaccard logic unchanged."""
        root = self.ensure_trie()
        return shortlist_by_trie(
            span_text,
            root=root,
            idf=self.idf,
            chain_grams=self.chain_grams,
            char_n=self.char_n,       # expects Optional[int] support in shortlist_by_trie
            token_ns=self.token_ns,
            topk=topk,
            tau=tau,
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe serialization for persistence next to lexicon/FAISS."""
        pack_posts = {g: {str(cid): int(c) for cid, c in cnts.items()} for g, cnts in self.postings.items()}
        return {
            "kind": "spacy_ngram_index",
            "version": 1,
            "token_ns": list(self.token_ns),
            "char_n": (self.char_n if self.char_n is not None else None),
            "idf": self.idf,
            "chain_grams": {str(k): dict(v) for k, v in self.chain_grams.items()},
            "postings": pack_posts,
        }

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "NGramIndex":
        """Rehydrate from JSON-safe dict."""
        token_ns = tuple(obj.get("token_ns") or (2, 3, 4))
        char_n = obj.get("char_n", None)
        idf = obj.get("idf") or {}
        # restore Counters
        chain_grams = {int(k): Counter(v) for k, v in (obj.get("chain_grams") or {}).items()}
        postings = {
            g: Counter({int(cid): int(c) for cid, c in cnts.items()})
            for g, cnts in (obj.get("postings") or {}).items()
        }
        return cls(token_ns=token_ns, char_n=char_n, idf=idf, chain_grams=chain_grams, postings=postings, root=None)

def build_ngram_index(
    chains: List[Dict[str, Any]],
    *,
    token_ns: Tuple[int, ...] = (2, 3, 4),
    char_n: Optional[int] = None,
    build_trie: bool = True,
) -> NGramIndex:
    """Single builder for both single-doc (in-memory) and cross-doc (persisted) modes."""
    df = Counter()
    chain_grams: Dict[int, Counter] = {}
    postings: Dict[str, Counter] = {}

    for ch in chains:
        cid = int(ch.get("chain_id"))
        grams = Counter()

        for m in ch.get("mentions", []) or []:
            txt = m.get("text", "") if isinstance(m, dict) else str(m)
            toks = _norm_tokens(txt)     # your spaCy-based normalizer
            heads = _head_lemmas(txt)

            if char_n is not None:
                grams.update(_char_ngrams(txt, char_n))
            for n in token_ns:
                grams.update(_token_ngrams(toks, n))
            grams.update(heads)

        if not grams:
            continue

        chain_grams[cid] = grams

        seen = set()
        for g, c in grams.items():
            postings.setdefault(g, Counter())[cid] += int(c)
            if g not in seen:
                df[g] += 1
                seen.add(g)

    N = max(len(chains), 1)
    idf = {g: math.log((N + 1) / (df[g] + 0.5)) for g in df}

    idx = NGramIndex(
        token_ns=token_ns,
        char_n=char_n,
        idf=idf,
        chain_grams=chain_grams,
        postings=postings,
        root=None,
    )
    if build_trie:
        idx.ensure_trie()
    return idx

import sqlite3

def ensure_documents_table(sqlite_path: str):
    """
    Make sure the DB has a `documents` table.
    If missing, create it and seed it from legacy tables (spans_file, files).
    """
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()

    # Create table if it doesn't exist
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
      doc_id      INTEGER PRIMARY KEY,
      uri         TEXT NOT NULL UNIQUE,
      created_at  TEXT,
      text_length INTEGER
    )""")

    # Seed from legacy spans_file
    try:
        cur.execute("INSERT OR IGNORE INTO documents (doc_id, uri) SELECT id, uri FROM spans_file")
    except Exception:
        pass

    # Seed from legacy files
    try:
        cur.execute("INSERT OR IGNORE INTO documents (doc_id, uri) SELECT id, path FROM files")
    except Exception:
        pass
    
    cols = {row[1] for row in cur.execute("PRAGMA table_info(documents)").fetchall()}

    if "full_text" not in cols:
        try:
            cur.execute("ALTER TABLE documents ADD COLUMN full_text TEXT")
        except Exception:
            pass

    if "text_length" not in cols:
        try:
            cur.execute("ALTER TABLE documents ADD COLUMN text_length INTEGER")
        except Exception:
            pass
    
    con.commit()
    con.close()


"""
Created on Sat Aug 16 17:04:29 2025

@author: niran
"""

