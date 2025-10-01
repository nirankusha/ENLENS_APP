# -*- coding: utf-8 -*-
"""
scico_graph_pipeline.py
-----------------------
SciCO graph builder with:
- coherence-based candidate shortlist (FAISS/LSH/optional coherence),
- selectable clustering ("auto" | "kmeans" | "torque" | "both" | "none"),
- community detection (greedy/louvain/leiden/labelprop),
- (NEW) cluster/community summarization:
    * centroid representative sentence
    * XSum-style extract via 'summarizer' (Derek Miller)
    * PreSumm top-scoring sentence, optionally SDG re-rank via CrossEncoder
"""

from __future__ import annotations
import re
import logging
from typing import List, Dict, Any, Tuple, Optional, Callable, Sequence
from dataclasses import dataclass, field
import hashlib
import numpy as np
import networkx as nx

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from sentence_transformers import SentenceTransformer

# coherence shortlist
from coherence_sampler import shortlist_by_coherence

# Optional libs
try:
    from sentence_transformers import CrossEncoder as HF_CrossEncoder
except Exception:
    HF_CrossEncoder = None

try:
    import community as community_louvain  # python-louvain
except Exception:
    community_louvain = None

try:
    import igraph as ig
    import leidenalg as la
except Exception:
    ig = la = None

# Optional summarizer (Derek Miller)
try:
    from summarizer import Summarizer as XSumSummarizer
except Exception:
    XSumSummarizer = None

# PreSumm helpers (we re-use your helpers from xsum_rank.py)
try:
    from xsum_rank import prepare_data_for_presum, batch_data
except Exception:
    prepare_data_for_presum = None
    batch_data = None

logger = logging.getLogger(__name__)

# ----------------------------- SciCo utilities -----------------------------
_DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"

def _hash_sentence(text: str) -> str:
    if text is None:
        text = ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _hash_sdg_targets(targets: Dict[str, Any]) -> str:
    if not targets:
        return "none"
    items = sorted((str(k), str(v)) for k, v in targets.items())
    payload = "|".join(f"{k}:{v}" for k, v in items)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _coerce_embedding_vector(vec: Any) -> Optional[np.ndarray]:
    if vec is None:
        return None
    if isinstance(vec, dict):
        for key in ("embedding", "vector", "values", "data"):
            if key in vec:
                vec = vec[key]
                break
        else:
            return None
    try:
        arr = np.asarray(vec, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    return None


def _maybe_int(value: Any) -> Any:
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    except Exception:
        pass
    return value


def _organize_precomputed_embeddings(precomputed: Any, n_sentences: int) -> Dict[str, Dict[Any, np.ndarray]]:
    info = {
        "model": None,
        "by_index": {},
        "by_sentence_id": {},
        "by_hash": {},
    }
    if precomputed is None:
        return info
    if isinstance(precomputed, dict):
        model = precomputed.get("model") or precomputed.get("model_name") or precomputed.get("backend")
        if isinstance(model, str):
            info["model"] = model

        direct_maps = {
            "by_index": precomputed.get("by_index"),
            "by_sentence_id": precomputed.get("by_sentence_id"),
            "by_hash": precomputed.get("by_hash"),
        }
        for key, mapping in direct_maps.items():
            if isinstance(mapping, dict):
                target = info[key]
                for k, vec in mapping.items():
                    arr = _coerce_embedding_vector(vec)
                    if arr is None:
                        continue
                    if key == "by_index":
                        idx = _maybe_int(k)
                        if isinstance(idx, int) and 0 <= idx < n_sentences:
                            target[idx] = arr
                    elif key == "by_sentence_id":
                        target[_maybe_int(k)] = arr
                    else:
                        target[str(k)] = arr

        # Support paired sentence_ids/vectors structure
        sent_ids = precomputed.get("sentence_ids")
        vectors = precomputed.get("vectors")
        if isinstance(sent_ids, (list, tuple)) and isinstance(vectors, (list, tuple)):
            for sid, vec in zip(sent_ids, vectors):
                arr = _coerce_embedding_vector(vec)
                if arr is not None:
                    info["by_sentence_id"][_maybe_int(sid)] = arr

        # Fallback: treat remaining keys as direct mappings
        reserved = {"model", "model_name", "backend", "by_index", "by_sentence_id", "by_hash", "sentence_ids", "vectors"}
        for key, value in precomputed.items():
            if key in reserved:
                continue
            arr = _coerce_embedding_vector(value)
            if arr is None:
                continue
            norm_key = _maybe_int(key)
            if isinstance(norm_key, int):
                if 0 <= norm_key < n_sentences:
                    info["by_index"][norm_key] = arr
                else:
                    info["by_sentence_id"][norm_key] = arr
            else:
                info["by_hash"][str(key)] = arr
    elif isinstance(precomputed, (list, tuple)):
        for idx, vec in enumerate(precomputed):
            if idx >= n_sentences:
                break
            arr = _coerce_embedding_vector(vec)
            if arr is not None:
                info["by_index"][idx] = arr
    return info


@dataclass
class GraphFeatureCache:
    embeddings: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    crossencoder: Dict[Tuple[str, str, str], Dict[str, Any]] = field(default_factory=dict)
    fingerprint: Optional[str] = None

    def invalidate(
        self,
        sentences: Optional[Sequence[str]] = None,
        *,
        embeddings: bool = True,
        crossencoder: bool = True,
    ) -> None:
        if sentences is None:
            if embeddings:
                self.embeddings.clear()
            if crossencoder:
                self.crossencoder.clear()
            return

        hashes = {_hash_sentence(s) for s in sentences}
        if embeddings:
            for key in list(self.embeddings.keys()):
                if key[1] in hashes:
                    self.embeddings.pop(key, None)
        if crossencoder:
            for key in list(self.crossencoder.keys()):
                if key[2] in hashes:
                    self.crossencoder.pop(key, None)

    def get_embedding(self, backend: str, sentence_hash: str) -> Optional[np.ndarray]:
        return self.embeddings.get((backend, sentence_hash))

    def set_embedding(self, backend: str, sentence_hash: str, vector: np.ndarray) -> None:
        if vector is None:
            return
        self.embeddings[(backend, sentence_hash)] = np.asarray(vector, dtype=np.float32)

    def get_crossencoder(self, model: str, sdg_hash: str, sentence_hash: str) -> Optional[Dict[str, Any]]:
        cached = self.crossencoder.get((model, sdg_hash, sentence_hash))
        if cached is None:
            return None
        return dict(cached)

    def set_crossencoder(self, model: str, sdg_hash: str, sentence_hash: str, scores: Dict[str, Any]) -> None:
        self.crossencoder[(model, sdg_hash, sentence_hash)] = dict(scores)


_DEFAULT_FEATURE_CACHE = GraphFeatureCache()


def get_default_feature_cache() -> GraphFeatureCache:
    return _DEFAULT_FEATURE_CACHE


def _select_embedding_backend(
    embedder: Optional[SentenceTransformer],
    scico_cfg: ScicoConfig,
    preferred_model: Optional[str] = None,
):
    if embedder is not None:
        name = getattr(embedder, "model_name", None) or getattr(embedder, "name", None)
        backend = f"custom::{name or embedder.__class__.__name__}"

        def encode(texts: List[str]) -> np.ndarray:
            return np.asarray(embedder.encode(texts), dtype=np.float32)

        return backend, encode

    model_hint = preferred_model
    try:
        from helper import sim_model as _default_embedder

        backend_name = getattr(_default_embedder, "model_name", None) or getattr(_default_embedder, "name", None)
        backend = f"helper::{backend_name or _default_embedder.__class__.__name__}"

        def encode(texts: List[str]) -> np.ndarray:
            return np.asarray(_default_embedder.encode(texts), dtype=np.float32)

        return backend, encode
    except Exception:
        pass

    backend = model_hint or _DEFAULT_EMBED_MODEL

    def encode(texts: List[str]) -> np.ndarray:
        return embed_sentences(texts, embedder=None, device=scico_cfg.device)

    return backend, encode

SCICO_LABELS = {
    0: "not_related",
    1: "corefer",
    2: "parent",   # m1 parent of m2
    3: "child"     # m1 child of m2
}

@dataclass
class ScicoConfig:
    model_name: str = "allenai/longformer-scico"
    device: str     = "cuda" if torch.cuda.is_available() else "cpu"
    prob_threshold: float = 0.5
    max_length: int = 4096

def load_scico(cfg: ScicoConfig = ScicoConfig()):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
    mdl.to(cfg.device).eval()
    # cache tokens for global attention
    start_token_id = tok.convert_tokens_to_ids("<m>")
    end_token_id   = tok.convert_tokens_to_ids("</m>")
    def build_global_attention(input_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(input_ids)
        mask[:, 0] = 1  # CLS / <s>
        starts = (input_ids == start_token_id).nonzero(as_tuple=False)
        ends   = (input_ids == end_token_id).nonzero(as_tuple=False)
        if starts.numel() or ends.numel():
            globs = torch.cat([x for x in (starts, ends) if x.numel()])
            mask.index_put_(tuple(globs.t()), torch.ones(globs.shape[0], dtype=mask.dtype, device=mask.device))
        return mask
    return tok, mdl, build_global_attention

def _mark_first(sentence: str, mention: str) -> Tuple[str, bool]:
    if not mention or not sentence:
        return sentence, False
    pat = re.compile(rf"(?i)\b{re.escape(mention)}\b")
    def repl(m): return f"<m>{m.group(0)}</m>"
    new, n = pat.subn(repl, sentence, count=1)
    if n == 0:
        idx = sentence.lower().find(mention.lower())
        if idx >= 0:
            new = sentence[:idx] + "<m>" + sentence[idx:idx+len(mention)] + "</m>" + sentence[idx+len(mention):]
            return new, True
        return sentence, False
    return new, True

@torch.no_grad()
def scico_pair_scores_batch(tok, mdl, build_gmask,
                            pairs, device: str, batch_size: int = 8):
    out_probs, out_labels = [], []
    for b in range(0, len(pairs), batch_size):
        chunk = pairs[b:b+batch_size]
        texts = []
        for (s1, m1, s2, m2) in chunk:
            ms1, _ = _mark_first(s1, m1)
            ms2, _ = _mark_first(s2, m2)
            texts.append(ms1 + " </s></s> " + ms2)
        enc = tok(texts, return_tensors="pt", truncation=True, max_length=4096, padding=True)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        gmask     = build_gmask(input_ids).to(device)
        logits = mdl(input_ids=input_ids, attention_mask=attn_mask, global_attention_mask=gmask).logits
        probs  = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        labs   = probs.argmax(axis=-1)
        out_probs.extend(list(probs))
        out_labels.extend(list(labs))
    return np.asarray(out_probs), np.asarray(out_labels)


# ----------------------------- Embeddings & clustering -----------------------------

def embed_sentences(sentences: List[str], embedder: Optional[SentenceTransformer]=None, device: Optional[str]=None) -> np.ndarray:
    if embedder is None:
        embedder = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
    if device is not None:
        embedder.to(device)
    embs = embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return embs

def cluster_kmeans(embeddings: np.ndarray, n_clusters: int = 5, seed: int = 0) -> np.ndarray:
    if len(embeddings) == 0:
        return np.array([])
    n_clusters = max(1, min(n_clusters, len(embeddings)))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = km.fit_predict(embeddings)
    return labels

def cluster_torque(embeddings: np.ndarray):
    try:
        from TorqueClustering import TorqueClustering
        DM = pairwise_distances(embeddings, embeddings, metric="euclidean")
        idx = TorqueClustering(DM, K=0, isnoise=False, isfig=False)[0]
        return np.array(idx)
    except Exception:
        return None


# ----------------------------- CrossEncoder features -----------------------------

def crossencoder_topk(
    sentences: List[str],
    sdg_targets: Optional[Dict[str, str]] = None,
    top_k: int = 3,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    *,
    cache: Optional[GraphFeatureCache] = None,
    sentence_hashes: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
    if sdg_targets is None or HF_CrossEncoder is None:
        return [{} for _ in sentences]
    
    cache = cache or _DEFAULT_FEATURE_CACHE
    hashes = sentence_hashes or [_hash_sentence(s) for s in sentences]
    sdg_hash = _hash_sdg_targets(sdg_targets)
    goals = list(sdg_targets.keys())
    results: List[Optional[Dict[str, Any]]] = [None] * len(sentences)
    missing: List[Tuple[int, str, str]] = []

    for idx, (sent, h) in enumerate(zip(sentences, hashes)):
        cached = cache.get_crossencoder(model_name, sdg_hash, h) if cache else None
        if cached is not None:
            results[idx] = cached
            continue
        missing.append((idx, sent, h))

    if missing:
        encoder = HF_CrossEncoder(model_name)
        for idx, sent, h in missing:
            pairs = [(sent, g) for g in goals]
            sc = encoder.predict(pairs)
            top = sorted(zip(goals, sc), key=lambda x: x[1], reverse=True)[:top_k]
            payload = {g: float(score) for g, score in top}
            results[idx] = payload
            if cache:
                cache.set_crossencoder(model_name, sdg_hash, h, payload)

    return [res or {} for res in results]


# ----------------------------- Community helpers -----------------------------

def _project_for_communities(G: nx.DiGraph, on: str = "all", weight_key: str = "prob") -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    def _ok(label: str) -> bool:
        if on == "all": return True
        if on == "corefer": return label == "corefer"
        if on == "parent_child": return label in {"parent", "child"}
        return True
    for u, v, d in G.edges(data=True):
        if not _ok(d.get("label")): continue
        w = float(d.get(weight_key, 1.0))
        if H.has_edge(u, v):
            H[u][v]["weight"] += w
        else:
            H.add_edge(u, v, weight=w)
    return H

def _run_communities(H: nx.Graph, method: str = "greedy", weight: Optional[str] = "weight") -> Dict[int, int]:
    if method == "none" or H.number_of_nodes() == 0:
        return {n: -1 for n in H.nodes()}
    if method == "greedy":
        comms = nx.algorithms.community.greedy_modularity_communities(H, weight=weight)
        return {n: cid for cid, cset in enumerate(comms) for n in cset}
    if method == "labelprop":
        comms = nx.algorithms.community.asyn_lpa_communities(H, weight=weight)
        return {n: cid for cid, cset in enumerate(comms) for n in cset}
    if method == "louvain" and community_louvain is not None:
        return community_louvain.best_partition(H, weight=weight)
    if method == "leiden" and ig is not None and la is not None:
        gi = ig.Graph()
        nodes = list(H.nodes())
        gi.add_vertices(nodes)
        idx = {n: i for i, n in enumerate(nodes)}
        gi.add_edges([(idx[u], idx[v]) for u, v in H.edges()])
        if weight and H.number_of_edges() > 0:
            gi.es[weight] = [float(H[u][v].get(weight, 1.0)) for u, v in H.edges()]
        part = la.find_partition(gi, la.CPMVertexPartition, weights=weight)
        mapping = {}
        for cid, comm in enumerate(part):
            for vid in comm:
                mapping[nodes[vid]] = cid
        return mapping
    # fallback
    return _run_communities(H, method="greedy", weight=weight)


# ----------------------------- Summarization utilities -----------------------------

def _centroid_representative(embeddings: np.ndarray, sentences: List[str], labels: np.ndarray, k: int) -> Dict[int, Dict[str, Any]]:
    """For each cluster id in 0..k-1: pick sentence nearest to centroid."""
    reps = {}
    if len(sentences) == 0: return reps
    # compute centroids
    centroids = np.vstack([embeddings[labels == i].mean(axis=0) if np.any(labels == i) else np.zeros(embeddings.shape[1]) for i in range(k)])
    closest_idxs, _ = pairwise_distances_argmin_min(centroids, embeddings, metric="euclidean")
    for cid in range(k):
        if not np.any(labels == cid): continue
        rep_idx = int(closest_idxs[cid])
        reps[cid] = {"representative": sentences[rep_idx], "representative_idx": rep_idx}
    return reps

def _xsum_summarize_group(sentences: List[str], num_sentences: int = 1) -> Optional[str]:
    if XSumSummarizer is None or not sentences:
        return None
    try:
        summ = XSumSummarizer()
        return summ(" ".join(sentences), num_sentences=num_sentences)
    except Exception:
        return None

def _presumm_top_sentence(sentences: List[str], model, tokenizer, device: str = None) -> Tuple[Optional[str], Optional[float]]:
    if prepare_data_for_presum is None or batch_data is None or model is None or tokenizer is None:
        return None, None
    try:
        instance = prepare_data_for_presum(sentences, tokenizer, max_len=512)
        src, segs, clss, mask_src, mask_cls = batch_data([instance])
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        src, segs, clss, mask_src = [t.to(dev) for t in (src, segs, clss, mask_src)]
        model.to(dev).eval()
        with torch.no_grad():
            # model.bert returns contextual embeddings [1, seq_len, h], PreSumm ext layer over CLS positions
            top_vec = model.bert(src, segs, mask_src)           # [1, L, H]
            cls_embs = top_vec[0, clss[0]]                      # [num_sents, H]
            wo = model.ext_layer.wo                             # nn.Linear(H->1)
            logits = cls_embs @ wo.weight.t() + wo.bias         # [num_sents,1]
            scores = torch.sigmoid(logits).squeeze(-1).detach().cpu().numpy()
        max_idx = int(np.argmax(scores))
        return sentences[max_idx], float(scores[max_idx])
    except Exception:
        return None, None

def _sdg_rerank(text: str, sdg_targets: Dict[str, str], top_k: int = 3, model_name: Optional[str] = None):
    if not text or sdg_targets is None or HF_CrossEncoder is None or model_name is None:
        return None
    try:
        goals = list(sdg_targets.keys())
        ce = HF_CrossEncoder(model_name)
        pairs = [(text, g) for g in goals]
        sc = ce.predict(pairs)
        ranked = sorted(zip(goals, sc), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"goal": g, "score": float(s)} for g, s in ranked]
    except Exception:
        return None


# ----------------------------- Graph building -----------------------------

def build_graph_from_selection(
    rows: List[Dict[str, Any]],
    *,
    selected_terms: List[str],
    sdg_targets: Optional[Dict[str, Any]] = None,
    kmeans_k: int = 5,
    use_torque: bool = False,                    # back-compat
    scico_cfg: Optional[ScicoConfig] = None,
    embedder: Optional[SentenceTransformer] = None,
    add_layout: bool = True,
    candidate_pairs: Optional[List[Tuple[int, int]]] = None,
    max_degree: int = 30,
    top_edges_per_node: int = 30,
    use_coherence_shortlist: bool = False,
    coherence_opts: Optional[Dict[str, Any]] = None,
    # clustering selection
    clustering_method: str = "auto",             # "auto" | "kmeans" | "torque" | "both" | "none"
    # communities
    community_on: str = "all",                   # "all" | "corefer" | "parent_child"
    community_method: str = "greedy",            # "greedy" | "louvain" | "leiden" | "labelprop" | "none"
    community_weight: str = "prob",
    # --- NEW summarization controls ---
    summarize: bool = False,                     # master switch
    summarize_on: str = "community",             # "community" | "kmeans" | "torque"
    summary_methods: Optional[List[str]] = None, # any of ["centroid","xsum","presumm"]
    summary_opts: Optional[Dict[str, Any]] = None,
    precomputed_embeddings: Optional[Any] = None,
    feature_cache: Optional[GraphFeatureCache] = None,
    cache_control: Optional[Dict[str, Any]] = None,
    ):
    """
    Build SciCO graph and (optionally) summarize clusters/communities.

    Summarization:
      summarize=True enables it.
      summarize_on selects which partition to summarize: communities or a clustering.
      summary_methods: choose any subset of ["centroid","xsum","presumm"].
      summary_opts:
        - for "xsum": {"num_sentences": 1}
        - for "presumm": {"presumm_model": ExtSummarizer, "presumm_tokenizer": BertTokenizer, "device": "cuda"}
        - for SDG re-rank: {"sdg_targets": {...}, "sdg_top_k": 3, "cross_encoder_model": "..."}
      Caching / reuse:
      precomputed_embeddings may supply vectors by index, sentence id or text hash.
      feature_cache caches embeddings + cross-encoder scores across invocations.
      cache_control supports {"invalidate": bool, "invalidate_sentences": [...], "fingerprint": "..."}.  
    """
    if scico_cfg is None:
        scico_cfg = ScicoConfig(prob_threshold=0.55)
    if summary_methods is None:
        summary_methods = []
    summary_opts = summary_opts or {}
    
    feature_cache = feature_cache or _DEFAULT_FEATURE_CACHE
    cache_control = cache_control or {}

    if cache_control.get("invalidate"):
        feature_cache.invalidate(
            embeddings=cache_control.get("invalidate_embeddings", True),
            crossencoder=cache_control.get("invalidate_crossencoder", True),
        )
    if cache_control.get("invalidate_sentences"):
        feature_cache.invalidate(
            cache_control.get("invalidate_sentences"),
            embeddings=cache_control.get("invalidate_embeddings", True),
            crossencoder=cache_control.get("invalidate_crossencoder", True),
        )
    if "fingerprint" in cache_control:
        fp = str(cache_control.get("fingerprint"))
        if feature_cache.fingerprint is not None and feature_cache.fingerprint != fp:
            feature_cache.invalidate()
        feature_cache.fingerprint = fp
    # ---------- 1) Collect texts ----------
    sentences = [r["text"] for r in rows]
    n = len(sentences)
    sentence_hashes = [_hash_sentence(s) for s in sentences]

    if n <= 1:
        G = nx.DiGraph()
        for i, r in enumerate(rows):
            G.add_node(i, text=r["text"], path=r.get("path"), start=r.get("start"), end=r.get("end"))
        meta = {"pairs_scored": 0, "communities": {}, "wcc": {}, "clustering_method": clustering_method}
        if summarize:
            meta["summaries"] = {}
        return G, meta

    # ---------- 2) Embeddings ----------
    precomp_info = _organize_precomputed_embeddings(precomputed_embeddings, n)
    backend_id, encode_fn = _select_embedding_backend(embedder, scico_cfg, preferred_model=precomp_info.get("model"))

    embeddings: List[Optional[np.ndarray]] = [None] * n
    missing: List[Tuple[int, str]] = []

    for idx, (row, text_hash) in enumerate(zip(rows, sentence_hashes)):
        # Inline embedding on the row (e.g., threaded from pipeline)
        row_emb = None
        for key in ("embedding", "vector", "mpnet_embedding"):
            if key in row and row[key] is not None:
                row_emb = _coerce_embedding_vector(row[key])
                if row_emb is not None:
                    break

        if row_emb is None:
            # Check organized precomputed pools
            row_emb = (
                precomp_info["by_index"].get(idx)
                or precomp_info["by_sentence_id"].get(_maybe_int(row.get("sid")))
                or precomp_info["by_sentence_id"].get(_maybe_int(row.get("sentence_id")))
                or precomp_info["by_hash"].get(text_hash)
            )

        if row_emb is not None:
            embeddings[idx] = row_emb
            feature_cache.set_embedding(backend_id, text_hash, row_emb)
            continue

        cached = feature_cache.get_embedding(backend_id, text_hash)
        if cached is not None:
            embeddings[idx] = cached
            continue

        missing.append((idx, text_hash))

    if missing:
        to_encode = [sentences[idx] for idx, _ in missing]
        new_embs = encode_fn(to_encode)
        new_arr = np.asarray(new_embs, dtype=np.float32)
        if new_arr.ndim == 1:
            new_arr = new_arr.reshape(1, -1)
        for (idx, text_hash), vec in zip(missing, new_arr):
            embeddings[idx] = vec
            feature_cache.set_embedding(backend_id, text_hash, vec)

    if any(e is None for e in embeddings):
        raise RuntimeError("Failed to obtain embeddings for all sentences")

    embs = np.vstack([np.asarray(e, dtype=np.float32) for e in embeddings])

    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    X = embs / norms  # normalized cosine/IP ready

    # ---------- 3) CrossEncoder features ----------
    ce_feats = crossencoder_topk(
        sentences,
        sdg_targets=sdg_targets,
        top_k=3,
        cache=feature_cache,
        sentence_hashes=sentence_hashes,
    )

    # ---------- 4) Clustering (SELECTABLE) ----------
    kml = None
    tql = None
    method = (clustering_method or "auto").lower()
    if method == "auto":
        kml = cluster_kmeans(embs, n_clusters=kmeans_k)
        if use_torque:
            tql = cluster_torque(embs)
    elif method == "kmeans":
        kml = cluster_kmeans(embs, n_clusters=kmeans_k)
    elif method == "torque":
        tql = cluster_torque(embs)
    elif method == "both":
        kml = cluster_kmeans(embs, n_clusters=kmeans_k)
        tql = cluster_torque(embs)
    elif method == "none":
        pass
    else:
        kml = cluster_kmeans(embs, n_clusters=kmeans_k)
        if use_torque:
            tql = cluster_torque(embs)

    # ---------- 5) Candidate pairs ----------
    if candidate_pairs is not None:
        pair_indices = [(int(i), int(j)) for (i, j) in candidate_pairs if 0 <= i < n and 0 <= j < n and i < j]
    elif use_coherence_shortlist:
        opts = coherence_opts or {}
        try:
            cand = shortlist_by_coherence(
                texts=sentences, embeddings=X,
                faiss_topk=opts.get("faiss_topk", 32),
                nprobe=opts.get("nprobe", 8),
                add_lsh=opts.get("add_lsh", True),
                lsh_threshold=opts.get("lsh_threshold", 0.8),
                minhash_k=opts.get("minhash_k", 5),
                cheap_len_ratio=opts.get("cheap_len_ratio", 0.25),
                cheap_jaccard=opts.get("cheap_jaccard", 0.08),
                use_coherence=opts.get("use_coherence", False),
                coherence_threshold=opts.get("coherence_threshold", 0.55),
                max_pairs=opts.get("max_pairs", None),
            )
            pair_indices = sorted(cand)
        except Exception:
            pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # ---------- 6) SciCO scoring ----------
    tok, mdl, build_gmask = load_scico(scico_cfg)
    sel_terms = [t for t in (selected_terms or []) if t and t.strip()]

    def pick_span(text: str) -> str:
        for t in sel_terms:
            if t.lower() in text.lower(): return t
        return sel_terms[0] if sel_terms else ""

    payload, spans_used = [], []
    for (i, j) in pair_indices:
        mi = pick_span(sentences[i]); mj = pick_span(sentences[j])
        payload.append((sentences[i], mi, sentences[j], mj))
        spans_used.append((mi, mj))

    probs, labs = scico_pair_scores_batch(tok, mdl, build_gmask, payload, device=scico_cfg.device, batch_size=8)

    # ---------- 7) Graph ----------
    G = nx.DiGraph()
    for i, r in enumerate(rows):
        attrs = dict(text=r["text"], path=r.get("path"), start=r.get("start"), end=r.get("end"), crossencoder=ce_feats[i])
        if kml is not None: attrs["kmeans"] = int(kml[i])
        if tql is not None:
            try: attrs["torque"] = int(tql[i])
            except Exception: pass
        G.add_node(i, **attrs)

    for ((i, j), p, lab, (mi, mj)) in zip(pair_indices, probs, labs, spans_used):
        lab = int(lab); conf = float(p[lab])
        if conf < scico_cfg.prob_threshold or lab <= 0: continue
        lab_name = SCICO_LABELS[lab]
        if lab_name == "corefer":
            G.add_edge(i, j, label=lab_name, prob=conf, term=mi or mj)
            G.add_edge(j, i, label=lab_name, prob=conf, term=mi or mj)
        elif lab_name == "parent":
            G.add_edge(i, j, label=lab_name, prob=conf, term=mi)
        elif lab_name == "child":
            G.add_edge(j, i, label=lab_name, prob=conf, term=mj)

    # ---------- 8) Sparsify ----------
    if max_degree is not None and top_edges_per_node is not None:
        for u in list(G.nodes()):
            out_edges = list(G.out_edges(u, data=True))
            if len(out_edges) > max_degree:
                out_edges.sort(key=lambda e: float(e[2].get("prob", 0.0)), reverse=True)
                for (_, v, _) in out_edges[top_edges_per_node:]:
                    if G.has_edge(u, v): G.remove_edge(u, v)

    # ---------- 9) Components + communities ----------
    component_id = {}
    for cid, comp in enumerate(nx.weakly_connected_components(G)):
        for n_ in comp: component_id[n_] = cid
    nx.set_node_attributes(G, component_id, name="wcc")

    H_com = _project_for_communities(G, on=community_on, weight_key=community_weight)
    communities = _run_communities(H_com, method=community_method, weight="weight")
    nx.set_node_attributes(G, communities, name="community")

    try:
        comm_sets = {}
        for n_, c_ in communities.items(): comm_sets.setdefault(c_, set()).add(n_)
        community_modularity = nx.algorithms.community.quality.modularity(H_com, comm_sets.values(), weight="weight")
    except Exception:
        community_modularity = None

    # ---------- 10) Optional summarization ----------
    summaries = {}
    if summarize:
        # Choose partition to summarize
        part_kind = (summarize_on or "community").lower()
        if part_kind == "community":
            # Build mapping cid -> indices
            part_labels = communities
            # stable label order
            uniq = sorted(set(part_labels.values()))
            label_to_indices = {cid: [i for i in range(n) if part_labels.get(i, -1) == cid] for cid in uniq}
        elif part_kind == "kmeans" and kml is not None:
            uniq = sorted(set(int(x) for x in kml))
            label_to_indices = {cid: [i for i, lab in enumerate(kml) if int(lab) == cid] for cid in uniq}
        elif part_kind == "torque" and tql is not None:
            uniq = sorted(set(int(x) for x in tql))
            label_to_indices = {cid: [i for i, lab in enumerate(tql) if int(lab) == cid] for cid in uniq}
        else:
            uniq = []
            label_to_indices = {}

        # Preload CrossEncoder for SDG re-rank if requested
        ce_model_name = summary_opts.get("cross_encoder_model")
        sdg_map = summary_opts.get("sdg_targets")
        sdg_top_k = int(summary_opts.get("sdg_top_k", 3))

        # For each group
        for cid in uniq:
            idxs = label_to_indices[cid]
            if not idxs: continue
            group_sents = [sentences[i] for i in idxs]
            group_embs  = X[idxs, :]

            summaries[cid] = {}

            # a) centroid representative
            if "centroid" in summary_methods:
                rep = _centroid_representative(group_embs, group_sents,
                                               labels=np.zeros(len(group_sents), dtype=int), # dummy one cluster
                                               k=1).get(0, {})
                if rep:
                    summaries[cid]["representative"] = rep.get("representative")
                    if sdg_map and ce_model_name:
                        summaries[cid]["representative_sdg"] = _sdg_rerank(rep.get("representative"), sdg_map, sdg_top_k, ce_model_name)

            # b) xsum (extractive) summary via Summarizer
            if "xsum" in summary_methods:
                xsum_txt = _xsum_summarize_group(group_sents, num_sentences=int(summary_opts.get("num_sentences", 1)))
                if xsum_txt:
                    summaries[cid]["xsum_summary"] = xsum_txt
                    if sdg_map and ce_model_name:
                        summaries[cid]["xsum_sdg"] = _sdg_rerank(xsum_txt, sdg_map, sdg_top_k, ce_model_name)

            # c) PreSumm top-scoring sentence
            if "presumm" in summary_methods:
                presumm_model = summary_opts.get("presumm_model")
                presumm_tok   = summary_opts.get("presumm_tokenizer")
                device_str    = summary_opts.get("device")
                top_sent, top_score = _presumm_top_sentence(group_sents, presumm_model, presumm_tok, device=device_str)
                if top_sent:
                    summaries[cid]["presumm_top_sent"] = top_sent
                    summaries[cid]["presumm_top_score"] = top_score
                    if sdg_map and ce_model_name:
                        summaries[cid]["presumm_sdg"] = _sdg_rerank(top_sent, sdg_map, sdg_top_k, ce_model_name)

    # ---------- 11) Optional layout ----------
    meta = {
        "embeddings": embs,
        "pairs_scored": len(pair_indices),
        "edges": [(u, v, d.get("label"), d.get("prob")) for (u, v, d) in G.edges(data=True)],
        "communities": communities,
        "wcc": component_id,
        "community_on": community_on,
        "community_method": community_method,
        "community_modularity": community_modularity,
        "clustering_method": method,
    }
    if kml is not None: meta["kmeans"] = kml
    if tql is not None: meta["torque"] = tql
    if summarize: meta["summaries"] = summaries

    if add_layout:
        # cosine sim → [0,1] weights
        S_sim = (X @ X.T).astype("float32")
        H = nx.Graph()
        for i in range(n):
            H.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                w = float((np.clip(S_sim[i, j], -1.0, 1.0) + 1.0) / 2.0)
                if w > 0: H.add_edge(i, j, weight=w)
        pos = nx.spring_layout(H, weight="weight", seed=42, dim=2)
        nx.set_node_attributes(G, pos, name="pos")
        meta["pos"] = pos

    return G, meta
