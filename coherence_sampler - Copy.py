# -*- coding: utf-8 -*-
"""
coherence_sampler.py
--------------------
Coherence-aware candidate shortlisting to mitigate O(n^2) pair explosion.

Pipeline:
1) ANN shortlist (FAISS top-k neighbours) on normalized sentence embeddings.
2) (optional) Random-Projection LSH bucket pairs.
3) (optional) Cheap filters (length ratio, Jaccard overlap).
4) (optional) SGNLP CoherenceMomentum model on concatenated 2-sentence docs,
   scored in both orders (s1+s2, s2+s1); keep max score.
5) Return candidate pair indices (i,j) with i<j.
"""

from __future__ import annotations
from typing import List, Tuple, Set, Optional
import numpy as np

# ANN + MinHash helpers
from ann_index import FaissIndex, build_span_lsh

# ---------- cheap text filters ----------
def _len_ratio_ok(a: str, b: str, min_ratio: float = 0.25) -> bool:
    la, lb = max(1, len(a)), max(1, len(b))
    r = min(la, lb) / max(la, lb)
    return r >= min_ratio

def _jaccard_words(a: str, b: str) -> float:
    wa = set(w for w in a.lower().split())
    wb = set(w for w in b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)

# ---------- Coherence model wrapper (lazy) ----------
class CoherenceMomentumScorer:
    _instance = None

    def __init__(self, cfg_url: str = "https://storage.googleapis.com/sgnlp-models/models/coherence_momentum/config.json",
                       wts_url: str = "https://storage.googleapis.com/sgnlp-models/models/coherence_momentum/pytorch_model.bin"):
        # Lazy import to avoid hard dependency unless used
        from sgnlp.models.coherence_momentum import (
            CoherenceMomentumModel, CoherenceMomentumConfig, CoherenceMomentumPreprocessor
        )
        self.config = CoherenceMomentumConfig.from_pretrained(cfg_url)
        self.model  = CoherenceMomentumModel.from_pretrained(wts_url, config=self.config)
        self.preproc= CoherenceMomentumPreprocessor(self.config.model_size, self.config.max_len)
        self.model.eval()

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = CoherenceMomentumScorer()
        return cls._instance

    def pair_score(self, s1: str, s2: str) -> float:
        """
        Score a pair by treating it as a 2-sentence mini-document.
        We evaluate both orders and keep the max (order-invariant heuristic).
        """
        t1 = self.preproc([f"{s1} {s2}"])
        t2 = self.preproc([f"{s2} {s1}"])
        s1s2 = float(self.model.get_main_score(t1["tokenized_texts"]).item())
        s2s1 = float(self.model.get_main_score(t2["tokenized_texts"]).item())
        return max(s1s2, s2s1)

# ---------- main shortlisting function ----------
def shortlist_by_coherence(
    texts: List[str],
    embeddings: Optional[np.ndarray] = None,
    faiss_topk: int = 32,
    nprobe: int = 8,
    add_lsh: bool = True,
    lsh_threshold: float = 0.8,
    minhash_k: int = 5,
    cheap_len_ratio: float = 0.25,
    cheap_jaccard: float = 0.08,
    use_coherence: bool = False,
    coherence_threshold: float = 0.55,
    max_pairs: Optional[int] = None
) -> Set[Tuple[int,int]]:
    """
    Returns a set of candidate pairs (i,j) with i<j.
    - If `use_coherence` True, runs SGNLP scorer to keep pairs with score >= threshold.
    """
    n = len(texts)
    if n <= 1:
        return set()

    # 1) Embeddings ANN (caller may pass precomputed embeddings normalized)
    cand_pairs = set()
    if embeddings is not None and len(embeddings) == n:
        X = np.asarray(embeddings, dtype="float32")
        try:
            dim = X.shape[1]
            nlist = max(1, n // 64)
            idx = FaissIndex(dim, nlist=nlist, nprobe=int(nprobe))
            idx.add(X)
            D, I = idx.search(X, topk=min(faiss_topk, n-1))
            for a, neighs in enumerate(I):
                for b in neighs:
                    if b < 0 or a == b:
                        continue
                    i, j = (a, b) if a < b else (b, a)
                    cand_pairs.add((i, j))
        except Exception:
            # Fallback: local band (keeps O(n*k))
            for i in range(n):
                for j in range(i+1, min(i+1+faiss_topk, n)):
                    cand_pairs.add((i, j))

    # 2) MinHash LSH on token shingles (union with ANN pairs)
    if add_lsh:
        items = [(i, texts[i]) for i in range(n)]
        try:
            lsh_pairs = build_span_lsh(items, threshold=lsh_threshold, num_perm=128, k=minhash_k)
            cand_pairs |= lsh_pairs
        except Exception:
            pass

    # 3) Cheap text-based filters
    def cheap_ok(i, j):
        if not _len_ratio_ok(texts[i], texts[j], cheap_len_ratio):
            return False
        if _jaccard_words(texts[i], texts[j]) < cheap_jaccard:
            return False
        return True

    filtered = {(i, j) for (i, j) in cand_pairs if cheap_ok(i, j)}
    cand_pairs = filtered

    # 4) Optional coherence scoring
    if use_coherence and cand_pairs:
        scorer = CoherenceMomentumScorer.get()
        keep = set()
        for (i, j) in cand_pairs:
            s = scorer.pair_score(texts[i], texts[j])
            if s >= coherence_threshold:
                keep.add((i, j))
        cand_pairs = keep

    # 5) Cap if needed
    if max_pairs is not None and len(cand_pairs) > max_pairs:
        cand_pairs = set(list(cand_pairs)[:max_pairs])

    return cand_pairs

"""
Created on Sat Aug 23 14:16:47 2025

@author: niran
"""

