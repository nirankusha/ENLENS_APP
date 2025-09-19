# -*- coding: utf-8 -*-
# === NEW: span grammer & scoring ===
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, Literal, Iterable, Callable 
import re, math
from collections import Counter, defaultdict
from helper_addons import _norm_tokens, _token_ngrams, _char_ngrams, _head_lemmas

def _char_ngrams_simple(text: str, n=4) -> list[str]:
    s = re.sub(r"\s+", " ", text.strip().lower())
    return [s[i:i+n] for i in range(max(0, len(s)-n+1))] if s else []

def _make_grams(text, *, char_ns=(4,), token_ns=(2,3)):
    grams = Counter()
    toks = _norm_tokens(text)
    for n in token_ns: grams.update(_token_ngrams(toks, n))   # '§' joiner
    for n in char_ns:  grams.update(_char_ngrams(text, n))    # bare char n-grams
    grams.update(_head_lemmas(text))                          # heads
    return grams


def _idf_weighted_jaccard(Gs: dict, Gc: dict, idf: dict) -> float:
    # Gs: span grams (keys), Gc: chain grams
    inter = 0.0; union = 0.0
    keys_s = set(Gs)
    keys_c = set(Gc)
    for g in (keys_s | keys_c):
        w = float(idf.get(g, 1.0))
        union += w
        if g in keys_s and g in keys_c:
            inter += w
    return (inter / union) if union > 0 else 0.0

def build_global_superroot(cx):
    """
    Returns:
      superroot: {gram: { (doc_id, chain_id): count }}
      idf:       {gram: idf_value}
      meta:      {doc_id: {'chain_ids': [..]}}  (optional, reserved)
    """
    from flexiconc_adapter import load_all_doc_tries
    docs = load_all_doc_tries(cx)  # {doc_id: {'trie':..., 'idf':..., 'chain_grams':...}}
    postings = defaultdict(lambda: defaultdict(int))
    df = Counter()
    for doc_id, blob in (docs or {}).items():
        cg = blob.get("chain_grams", {})
        for chain_id, grams in cg.items():
            # df: gram appears in this (doc,chain)
            for g, c in grams.items():
                postings[g][(doc_id, int(chain_id))] += int(c)
            for g in grams.keys():
                df[g] += 1
    # idf
    N = max(1, sum(len(v) for v in (doc.get("chain_grams", {}) for doc in docs.values())))
    idf = {g: math.log((N + 1) / (df[g] + 1)) + 1.0 for g in df}
    return postings, idf, {}

def global_trie_shortlist(span_text: str, cx,
                          *, char_ns=(4,), token_ns=(2,3),
                          topk=10, tau=0.18) -> List[Dict]:
    """
    Returns [{doc_id, chain_id, score_trie, why='trie'}] ranked.
    """
    from flexiconc_adapter import load_all_doc_tries
    postings, idf, _ = build_global_superroot(cx)
    Gr = _make_grams(span_text, char_ns=char_ns, token_ns=token_ns)
    # collect candidate (doc,chain) from grams that appear in span
    cand = set()
    for g in Gr.keys():
        pl = postings.get(g)
        if pl:
            cand.update(pl.keys())

    if not cand:
        return []

    # score each candidate via IDF-weighted Jaccard
    docs = load_all_doc_tries(cx)
    out = []
    for (doc_id, chain_id) in cand:
        cg = docs.get(doc_id, {}).get("chain_grams", {}).get(str(chain_id)) \
             or docs.get(doc_id, {}).get("chain_grams", {}).get(int(chain_id)) \
             or {}
        s = _idf_weighted_jaccard(Gr, cg, idf)
        if s >= float(tau):
            out.append({"doc_id": doc_id, "chain_id": int(chain_id),
                        "score_trie": float(s), "why": "trie"})
    out.sort(key=lambda r: r["score_trie"], reverse=True)
    return out[:topk]

# ---- (optional) fuse with co-occ at corpus level ----
def global_coref_query_trie(span_text: str, cx,
                       *, use_trie=True, use_cooc=False,
                       topk=10, tau_trie=0.18, **unused) -> List[Dict]:
    """
    A thin orchestrator so SpanBERT code can just call one function.
    Currently returns trie-only ranking if use_cooc=False.
    """
    hits = []
    if use_trie:
        hits = global_trie_shortlist(span_text, cx, topk=topk, tau=tau_trie)
    # (you can extend here to add co-occ fusion later)
    # normalize shape: score = score_trie for now
    for h in hits:
        h.setdefault("score", h.get("score_trie", 0.0))
        h.setdefault("score_cooc", 0.0)
        h.setdefault("why", "trie")
    return hits[:topk]

# === NEW: chain vector from doc cooc (mean of row vectors for chain tokens) ===
def _chain_vector_from_doc_cooc(chain_lexicon: dict, vocab: dict, rows, norms) -> dict:
    """
    chain_lexicon: {gram: count} or tokens list; we'll use tokens (tok:*) only
    returns sparse dict col->weight (L2 normalized)
    """
    acc = defaultdict(float)
    # take token grams only
    toks = [g for g in chain_lexicon.keys() if "§" in g or g.isalpha()]
    idxs = [vocab.get(t) for t in toks if vocab.get(t) is not None]
    for i in idxs:
        # add row i (neighbors) into accumulator
        start, end = rows.indptr[i], rows.indptr[i+1]
        cols = rows.indices[start:end]; vals = rows.data[start:end]
        for j, v in zip(cols, vals):
            acc[j] += v
    # L2 normalize
    n = math.sqrt(sum(v*v for v in acc.values())) or 1.0
    for k in list(acc.keys()):
        acc[k] /= n
    return acc

def _cos_sparse(a: dict, b: dict) -> float:
    # assume a,b are L2 normalized dicts (col -> weight)
    if len(a) > len(b): a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())

# === main global coref query ===
def global_coref_query(span_text: str, cx,
                       *, use_trie=True, use_cooc=True,
                       topk=10, tau_trie=0.18, tau_cooc=0.18,
                       use_pair_scorer=False, scorer=None, scorer_tau=0.25):
    """
    Return list of dicts:
      {'doc_id', 'chain_id', 'score_trie', 'score_cooc', 'score', 'why': 'trie|cooc|both'}
    """
    from flexiconc_adapter import load_all_doc_tries, load_all_doc_coocs
    Gs = _make_grams(span_text, char_ns=(4,), token_ns=(2,3))

    results = {}
    # ---- TRIE shortlist over corpus ----
    postings, idf_global, _meta = build_global_superroot(cx) if use_trie else ({}, {}, {})
    if use_trie:
        # score each posting list entry
        for g in Gs.keys():
            plist = postings.get(g)
            if not plist: continue
            # accumulate candidates; scoring later
            for (doc_id, chain_id), _cnt in plist.items():
                results.setdefault((doc_id, chain_id), {"doc_id": doc_id, "chain_id": chain_id,
                                                        "score_trie": 0.0, "score_cooc": 0.0})

        # compute IDF-Jaccard per candidate
        docs = load_all_doc_tries(cx)
        for (doc_id, chain_id), row in list(results.items()):
            cg = docs.get(doc_id, {}).get("chain_grams", {}).get(str(chain_id)) \
                 or docs.get(doc_id, {}).get("chain_grams", {}).get(int(chain_id)) \
                 or {}
            s = _idf_weighted_jaccard(Gs, cg, idf_global)
            row["score_trie"] = s
        # prune by tau_trie
        results = {k:v for k,v in results.items() if v["score_trie"] >= tau_trie} or results

    # ---- COOC shortlist & scoring ----
    if use_cooc:
        coocs = load_all_doc_coocs(cx)  # {doc_id: (vocab, csr, norms)}
        # build span vector per-doc and compare to chain vectors
        for doc_id, (vocab, rows, norms) in coocs.items():
            # make span vector in this doc's cooc space
            # simple: average rows of last few tokens appearing in vocab
            span_tokens = _norm_tokens(span_text)
            if not span_tokens: continue
            acc = defaultdict(float)
            for t in span_tokens:
                i = vocab[t]
                start, end = rows.indptr[i], rows.indptr[i+1]
                cols = rows.indices[start:end]; vals = rows.data[start:end]
                for j, v in zip(cols, vals):
                    acc[j] += v
            # L2 normalize
            n = math.sqrt(sum(v*v for v in acc.values())) or 1.0
            for k in list(acc.keys()):
                acc[k] /= n

            # compare to each chain in this doc
            try_blob = load_all_doc_tries(cx).get(doc_id, {})
            chain_grams = try_blob.get("chain_grams", {})
            for chain_id, cg in chain_grams.items():
                ch_vec = _chain_vector_from_doc_cooc(cg, vocab, rows, norms)
                if not ch_vec: continue
                cos = _cos_sparse(acc, ch_vec)
                if cos >= tau_cooc:
                    row = results.setdefault((doc_id, int(chain_id)),
                                             {"doc_id": doc_id, "chain_id": int(chain_id),
                                              "score_trie": 0.0, "score_cooc": 0.0})
                    row["score_cooc"] = max(row["score_cooc"], float(cos))

    # ---- fuse & rank ----
    fused = []
    for row in results.values():
        wA = 0.6 if use_trie else 0.0
        wB = 0.4 if use_cooc else 0.0
        score = wA * row["score_trie"] + wB * row["score_cooc"]
        why = ("both" if (row["score_trie"]>0 and row["score_cooc"]>0)
               else "trie" if row["score_trie"]>0 else "cooc")
        fused.append({**row, "score": score, "why": why})
    fused.sort(key=lambda r: r["score"], reverse=True)
    return fused[:topk]

"""
Created on Fri Sep 12 14:05:18 2025

@author: niran
"""

