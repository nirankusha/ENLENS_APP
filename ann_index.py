# -*- coding: utf-8 -*-
# ann_index.py
import numpy as np

try:
    import faiss
except Exception:
    faiss = None

class FaissIndex:
    """
    Lightweight FAISS wrapper (IVF-Flat with inner product).
    Callers must pass L2-normalized vectors for cosine-like similarity.
    """
    def __init__(self, dim, nlist=64, nprobe=8):
        if faiss is None:
            raise ImportError("faiss is not installed. Use: pip install faiss-cpu")
        self.dim = dim
        self.nlist, self.nprobe = int(nlist), int(nprobe)
        self.quant = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(self.quant, dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
        self.trained = False

    def add(self, X: np.ndarray):
        if not self.trained:
            self.index.train(X)
            self.trained = True
            self.index.nprobe = self.nprobe
        self.index.add(X)

    def search(self, Q: np.ndarray, topk=32):
        return self.index.search(Q, int(topk))


# ---- MinHash helpers for span/mention strings ----
from datasketch import MinHash, MinHashLSH

def span_minhash(s: str, num_perm=128, k=5):
    m = MinHash(num_perm=num_perm)
    t = s.lower()
    for i in range(max(1, len(t) - k + 1)):
        m.update(t[i:i+k].encode('utf8'))
    return m

def build_span_lsh(items, threshold=0.8, num_perm=128, k=5):
    """
    items: list[(key, string)]
    returns: set of candidate pairs {(key_i, key_j), ...}
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mh_map = {}
    for key, s in items:
        mh = span_minhash(s, num_perm=num_perm, k=k)
        lsh.insert(key, mh)
        mh_map[key] = mh
    pairs = set()
    for key, mh in mh_map.items():
        for cand in lsh.query(mh):
            if cand == key: continue
            i, j = sorted((key, cand))
            pairs.add((i, j))
    return pairs
"""
Created on Sat Aug 23 12:26:37 2025

@author: niran
"""

