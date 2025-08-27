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
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

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


# ----------------------------- SciCo utilities -----------------------------

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

def crossencoder_topk(sentences: List[str],
                      sdg_targets: Optional[Dict[str,str]] = None,
                      top_k: int = 3,
                      model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> List[Dict[str, Any]]:
    if sdg_targets is None or HF_CrossEncoder is None:
        return [{} for _ in sentences]
    goals = list(sdg_targets.keys())
    encoder = HF_CrossEncoder(model_name)
    feats = []
    for s in sentences:
        pairs = [(s, g) for g in goals]
        sc    = encoder.predict(pairs)
        top   = sorted(zip(goals, sc), key=lambda x: x[1], reverse=True)[:top_k]
        feats.append({g: float(score) for g, score in top})
    return feats


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
    summary_opts: Optional[Dict[str, Any]] = None
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
    """
    if scico_cfg is None:
        scico_cfg = ScicoConfig(prob_threshold=0.55)
    if summary_methods is None:
        summary_methods = []
    summary_opts = summary_opts or {}

    # ---------- 1) Collect texts ----------
    sentences = [r["text"] for r in rows]
    n = len(sentences)

    if n <= 1:
        G = nx.DiGraph()
        for i, r in enumerate(rows):
            G.add_node(i, text=r["text"], path=r.get("path"), start=r.get("start"), end=r.get("end"))
        meta = {"pairs_scored": 0, "communities": {}, "wcc": {}, "clustering_method": clustering_method}
        if summarize:
            meta["summaries"] = {}
        return G, meta

    # ---------- 2) Embeddings ----------
    if embedder is None:
        try:
            from helper import sim_model as _default_embedder
            embedder = _default_embedder
            embs = np.asarray(embedder.encode(sentences), dtype="float32")
        except Exception:
            embs = embed_sentences(sentences, embedder=None, device=scico_cfg.device)
    else:
        embs = np.asarray(embedder.encode(sentences), dtype="float32")

    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    X = embs / norms  # normalized cosine/IP ready

    # ---------- 3) CrossEncoder features ----------
    ce_feats = crossencoder_topk(sentences, sdg_targets=sdg_targets, top_k=3)

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
