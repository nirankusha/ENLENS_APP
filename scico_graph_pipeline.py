
"""
scico_graph_pipeline.py
-----------------------
End-to-end pipeline to turn selected concordances (from flexiconc_overlay)
into a SciCo-linked, dual-clustered sentence graph with rich node features.

What it does
1) Takes a list of selected concordance rows: [{'sid', 'text', 'path', 'start', 'end'}, ...].
2) Computes pairwise *hierarchical cross-document coreference* links using
   allenai/longformer-scico (4-class: not-related, corefer, parent, child).
   Mentions are highlighted by wrapping the selected term in <m>...</m> in each sentence
   (as required by the model). Global attention is given to <s>, <m>, </m>. 
3) Embeds sentences with an ST model (defaults to 'paraphrase-mpnet-base-v2') and
   - (A) KMeans clustering
   - (B) Optional TorqueClustering (if installed)
4) Optionally computes CrossEncoder features per sentence for SDG targets (top‑k).
5) Builds a NetworkX graph whose nodes are sentences and whose edges are SciCo links.
   Adds a spring layout driven by cosine similarity between embeddings.

Dependencies
- transformers, sentence_transformers, networkx, numpy, scikit-learn
- (optional) TorqueClustering
- (optional) cross-encoder (HuggingFace CrossEncoder)

Integration points
- SDG targets can be provided (dict code -> description). If None, skips that feature.
- You can pass in your own SentenceTransformer instance via `embedder=`.
- Works smoothly with helper.sim_model if you import and pass it in.
"""

from __future__ import annotations
import re
import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import networkx as nx

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer, util

try:
    from sentence_transformers import CrossEncoder as HF_CrossEncoder
except Exception:
    HF_CrossEncoder = None

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
    prob_threshold: float = 0.5    # minimum softmax prob to create an edge
    max_length: int = 4096

def load_scico(cfg: ScicoConfig = ScicoConfig()):
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
    mdl.to(cfg.device).eval()
    # cache tokens for global attention
    start_token_id = tok.convert_tokens_to_ids("<m>")
    end_token_id   = tok.convert_tokens_to_ids("</m>")
    cls_index      = 0  # Longformer uses 0 for <s> aka [CLS]
    def build_global_attention(input_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(input_ids)
        mask[:, 0] = 1  # CLS / <s>
        # positions of <m> and </m>
        starts = (input_ids == start_token_id).nonzero(as_tuple=False)
        ends   = (input_ids == end_token_id).nonzero(as_tuple=False)
        if starts.numel() or ends.numel():
            globs = torch.cat([x for x in (starts, ends) if x.numel()])
            mask.index_put_(tuple(globs.t()), torch.ones(globs.shape[0], dtype=mask.dtype, device=mask.device))
        return mask
    return tok, mdl, build_global_attention

def _mark_first(sentence: str, mention: str) -> Tuple[str, bool]:
    """
    Wrap the first exact (case-insensitive, word-boundary) match of `mention`
    with <m>...</m>. Returns (new_sentence, found).
    """
    if not mention or not sentence:
        return sentence, False
    # word-boundary, escape special chars
    pat = re.compile(rf"(?i)\b{re.escape(mention)}\b")
    def repl(m):
        return f"<m>{m.group(0)}</m>"
    new, n = pat.subn(repl, sentence, count=1)
    if n == 0:
        # fallback: simple substring (no word boundaries)
        idx = sentence.lower().find(mention.lower())
        if idx >= 0:
            new = sentence[:idx] + "<m>" + sentence[idx:idx+len(mention)] + "</m>" + sentence[idx+len(mention):]
            return new, True
        return sentence, False
    return new, True

def scico_pair_scores(tok, mdl, build_gmask, sent1: str, span1: str, sent2: str, span2: str, device: str) -> Tuple[np.ndarray, int]:
    """Return (probs[4], argmax) for the pair after marking mentions and building global attention."""
    m1, f1 = _mark_first(sent1, span1)
    m2, f2 = _mark_first(sent2, span2)
    # if either not found, we still run without markers (model will default to CLS-only global attention)
    inputs = m1 + " </s></s> " + m2
    enc = tok(inputs, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    gmask = build_gmask(input_ids).to(device)
    with torch.no_grad():
        out = mdl(input_ids=input_ids, attention_mask=attn_mask, global_attention_mask=gmask)
        probs = torch.softmax(out.logits, dim=-1).squeeze(0).detach().cpu().numpy()
    return probs, int(probs.argmax())

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

# ----------------------------- Graph building -----------------------------

def build_graph_from_selection(rows: List[Dict[str, Any]],
                               selected_terms: List[str],
                               sdg_targets: Optional[Dict[str,str]] = None,
                               kmeans_k: int = 5,
                               use_torque: bool = False,
                               scico_cfg: ScicoConfig = ScicoConfig(),
                               embedder: Optional[SentenceTransformer] = None,
                               add_layout: bool = True):
    """
    rows: [{'sid','text','path','start','end'}, ...] typically from flexiconc_overlay state.matched_rows
    selected_terms: list of strings chosen in the overlay UI.
    Returns: (G, meta) where G is a NetworkX graph and meta includes embeddings, labels, etc.
    """
    # 1) collect sentences (dedupe by text+path range to be safe)
    seen = set()
    sentences, meta_rows = [], []
    for r in rows:
        key = (r.get("text","").strip(), r.get("path"), r.get("start"), r.get("end"))
        if key in seen: 
            continue
        seen.add(key)
        sentences.append(r["text"])
        meta_rows.append(r)

    if not sentences:
        return nx.Graph(), {"embeddings": np.zeros((0,)), "kmeans": np.array([]), "torque": None}

    # 2) embeddings + clustering
    embs = embed_sentences(sentences, embedder=embedder, device=scico_cfg.device)
    kml  = cluster_kmeans(embs, n_clusters=kmeans_k)
    tql  = cluster_torque(embs) if use_torque else None

    # 3) cross-encoder features (optional)
    ce_feats = crossencoder_topk(sentences, sdg_targets=sdg_targets, top_k=3)

    # 4) SciCo pairwise links per selected term
    tok, mdl, build_gmask = load_scico(scico_cfg)
    edges = []  # (i,j,label,prob,term)
    n = len(sentences)
    selected_terms = [t for t in selected_terms if t and t.strip()]
    for term in selected_terms:
        # find which sentences contain this term
        holders = [i for i,s in enumerate(sentences) if term.lower() in s.lower()]
        for a_idx in range(len(holders)):
            for b_idx in range(a_idx+1, len(holders)):
                i, j = holders[a_idx], holders[b_idx]
                probs, lab = scico_pair_scores(tok, mdl, build_gmask,
                                               sentences[i], term,
                                               sentences[j], term,
                                               device=scico_cfg.device)
                if float(probs[lab]) >= scico_cfg.prob_threshold and lab > 0:
                    edges.append((i, j, SCICO_LABELS[lab], float(probs[lab]), term))

    # 5) Build graph
    G = nx.DiGraph()
    for i, (s, row) in enumerate(zip(sentences, meta_rows)):
        G.add_node(i,
                   text=s,
                   path=row.get("path"),
                   start=row.get("start"),
                   end=row.get("end"),
                   kmeans=int(kml[i]),
                   torque=(int(tql[i]) if tql is not None else None),
                   crossencoder=ce_feats[i])
    for (i, j, lab, prob, term) in edges:
        # direction for parent/child; undirected for corefer
        if lab == "corefer":
            G.add_edge(i, j, label=lab, prob=prob, term=term)
            G.add_edge(j, i, label=lab, prob=prob, term=term)
        elif lab == "parent":
            # m1 parent of m2 -> i -> j
            G.add_edge(i, j, label=lab, prob=prob, term=term)
        elif lab == "child":
            # m1 child of m2 -> i <- j  (edge from j to i)
            G.add_edge(j, i, label=lab, prob=prob, term=term)

    # 6) Optional layout using cosine similarities
    meta = {"embeddings": embs, "kmeans": kml, "torque": tql, "edges": edges}
    if add_layout:
        # turn cosine sim into distances; build spring layout
        S = np.clip(util.cos_sim(embs, embs).cpu().numpy(), -1.0, 1.0)
        # weights for spring: higher similarity -> stronger attraction
        # networkx spring_layout uses 'weight' on edges; we create a dense aux graph
        H = nx.Graph()
        for i in range(n):
            H.add_node(i)
        for i in range(n):
            for j in range(i+1, n):
                w = float((S[i, j] + 1.0) / 2.0)  # map [-1,1] -> [0,1]
                if w > 0:
                    H.add_edge(i, j, weight=w)
        pos = nx.spring_layout(H, weight="weight", seed=42, dim=2)
        nx.set_node_attributes(G, pos, name="pos")
        meta["pos"] = pos

    return G, meta
