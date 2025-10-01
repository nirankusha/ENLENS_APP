# -*- coding: utf-8 -*-
"""
run_scico_graph.py
CLI + Streamlit helpers:
- Build SciCo graph with FAISS/MinHash-LSH shortlist (+ optional coherence)
- Save interactive HTML (PyVis) and JSON meta
- Render in Streamlit via components.html

Usage (CLI):
  python run_scico_graph.py "/path/to/file.pdf" \
    --terms poverty inequality \
    --use_shortlist --faiss_topk 32 --add_lsh --lsh_threshold 0.8 \
    --summarize --summary_methods centroid xsum \
    --html scico_graph.html --json scico_meta.json

In Streamlit:
  from run_scico_graph import build_scico_from_file, pyvis_html, clean_meta_for_json
  G, meta = build_scico_from_file("file.pdf", terms=["poverty"], use_shortlist=True)
  html = pyvis_html(G, meta, height_px=800)
  st.components.v1.html(html, height=820, scrolling=True)
  st.json(clean_meta_for_json(meta))
"""

import argparse, json, os
from pathlib import Path

import numpy as np
import networkx as nx

from pyvis.network import Network

from scico_graph_pipeline import build_graph_from_selection, ScicoConfig
from ENLENS_SpanBert_corefree_prod import run_quick_analysis


# -------------------------- JSON-safety helpers --------------------------

def _py_scalar(x):
    """Convert numpy scalars to native Python scalars for JSON."""
    import numpy as _np
    if isinstance(x, (_np.generic,)):
        return x.item()
    return x

def _clean_recursive(obj):
    """Recursively ensure JSON-native types."""
    import numpy as _np
    if isinstance(obj, dict):
        return {str(_py_scalar(k)): _clean_recursive(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_recursive(v) for v in obj]
    if isinstance(obj, (_np.generic,)):
        return obj.item()
    return obj

def clean_meta_for_json(meta: dict):
    """Return a JSON-safe subset of meta."""
    meta = meta or {}
    comm = meta.get("communities", {}) or {}
    sums = meta.get("summaries", {}) or {}
    out = {
        "pairs_scored": int(_py_scalar(meta.get("pairs_scored", 0))),
        "communities": {str(_py_scalar(k)): int(_py_scalar(v)) for k, v in comm.items()},
        "summaries": _clean_recursive(sums),
    }
    return out


# -------------------------- PyVis visualization --------------------------

_VIZ_OPTIONS = {
    "nodes": {"shape": "dot", "size": 6, "font": {"size": 12}},
    "edges": {
        "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
        "smooth": {"enabled": False},
        "color": {"opacity": 0.6},
    },
    "interaction": {
        "hover": True,
        "multiselect": True,
        "dragNodes": True,
        "hideEdgesOnDrag": True,
    },
    "physics": {
        "solver": "forceAtlas2Based",
        "stabilization": {"enabled": True, "iterations": 120},
        "forceAtlas2Based": {
            "gravitationalConstant": -50,
            "springLength": 90,
            "avoidOverlap": 0.6,
        },
    },
}

def _edge_color(lbl: str):
    return {"corefer": "#3a6", "parent": "#06c", "child": "#c60"}.get(lbl, "#888")

def _rows_from_production_output(prod):
    rows = []
    for s in prod.get("sentence_analyses", []) or []:
        rows.append(dict(
            sid=s.get("sentence_id"),
            text=s.get("sentence_text"),
            path="",
            start=s.get("doc_start"),
            end=s.get("doc_end"),
        ))
    return rows

def _safe_num(x):
    import numpy as _np
    if isinstance(x, (_np.generic,)):
        return x.item()
    return x

def pyvis_html(G: nx.DiGraph, meta: dict, title="SciCo graph", height_px: int = 800) -> str:
    """
    Build and return the HTML string for embedding (e.g., Streamlit components.html).
    """
    import json as _json

    # Layout
    pos = meta.get("pos")
    if not pos:
        pos = nx.spring_layout(G, seed=42, dim=2)

    xs = np.array([pos[n][0] for n in G.nodes()])
    ys = np.array([pos[n][1] for n in G.nodes()])
    sx = (xs - xs.mean()) / (xs.std() + 1e-8) * 400
    sy = (ys - ys.mean()) / (ys.std() + 1e-8) * 400

    nodes_list = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes_list)}

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#ffffff", font_color="#333", notebook=False)
    net.barnes_hut()

    # Nodes
    for i, data in G.nodes(data=True):
        i_py = _safe_num(i)
        label = (data.get("text", "") or "")[:140].replace("\n", " ")
        tooltip = data.get("text", "") or ""
        kmeans  = data.get("kmeans")
        torque  = data.get("torque")
        title_html = "<b>Sentence</b><br>" + tooltip
        extras = []
        if kmeans is not None: extras.append(f"kmeans={_safe_num(kmeans)}")
        if torque is not None: extras.append(f"torque={_safe_num(torque)}")
        if extras: title_html += "<br><i>" + ", ".join(extras) + "</i>"

        net.add_node(
            n_id=i_py,
            label=str(label),
            title=title_html,
            x=float(sx[node_index[i]]),
            y=float(sy[node_index[i]]),
        )

    # Edges
    for u, v, d in G.edges(data=True):
        u_py, v_py = _safe_num(u), _safe_num(v)
        lbl  = d.get("label", "rel")
        prob = float(d.get("prob", 0.0))
        term = d.get("term", "")
        title_e = f"{lbl} (p={prob:.2f})" + (f" — {term}" if term else "")
        net.add_edge(u_py, v_py, color=_edge_color(lbl), title=title_e, width=float(1 + 3*prob), arrows="to")

    net.set_options(_json.dumps(_VIZ_OPTIONS))

    # Return HTML as string (avoid notebook path issues)
    return net.generate_html(notebook=False)


def build_scico_from_file(
    input_path: str,
    max_sentences: int = 40,
    terms=None,
    *,
    # shortlist
    use_shortlist: bool = False,
    faiss_topk: int = 32,
    nprobe: int = 8,
    add_lsh: bool = False,
    lsh_threshold: float = 0.80,
    minhash_k: int = 5,
    cheap_len_ratio: float = 0.25,
    cheap_jaccard: float = 0.08,
    use_coherence: bool = False,
    coherence_threshold: float = 0.55,
    # clustering / communities
    clustering: str = "auto",          # auto|kmeans|torque|both|none
    kmeans_k: int = 5,
    community_on: str = "all",         # all|corefer|parent_child
    community_method: str = "greedy",  # greedy|louvain|leiden|labelprop|none
    prob_threshold: float = 0.55,
    # summaries
    summarize: bool = False,
    summarize_on: str = "community",   # community|kmeans|torque
    summary_methods=None,              # ["centroid","xsum","presumm"]
    xsum_sentences: int = 1,
    sdg_topk: int = 3,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
):
    """
    Programmatic (non-CLI) builder for use in Streamlit / notebooks.

    Returns:
        G (nx.DiGraph), meta (dict)
    """
    if summary_methods is None:
        summary_methods = ["centroid"]

    # 1) Sentence extraction (SpanBERT quick analysis)
    production_output = run_quick_analysis(input_path, max_sentences)
    rows = _rows_from_production_output(production_output)
    if len(rows) < 2:
        return nx.DiGraph(), {"pairs_scored": 0}
    
    cluster_analysis = production_output.get("cluster_analysis") if isinstance(production_output, dict) else None
    if isinstance(cluster_analysis, dict):
        precomputed_embeddings = cluster_analysis.get("embeddings")
    else:
        precomputed_embeddings = None
    # 2) Pipeline knobs
    coherence_opts = dict(
        faiss_topk=faiss_topk,
        nprobe=nprobe,
        add_lsh=add_lsh,
        lsh_threshold=lsh_threshold,
        minhash_k=minhash_k,
        cheap_len_ratio=cheap_len_ratio,
        cheap_jaccard=cheap_jaccard,
        use_coherence=use_coherence,
        coherence_threshold=coherence_threshold,
        max_pairs=None,
    )

    scico_cfg = ScicoConfig(prob_threshold=prob_threshold)

    # 3) Build graph
    G, meta = build_graph_from_selection(
        rows,
        selected_terms=terms or [],
        kmeans_k=kmeans_k,
        clustering_method=clustering,
        community_on=community_on,
        community_method=community_method,
        community_weight="prob",
        scico_cfg=scico_cfg,
        add_layout=True,
        # shortlist
        use_coherence_shortlist=use_shortlist,
        coherence_opts=coherence_opts,
        # sparsify defaults
        max_degree=30,
        top_edges_per_node=30,
        # summaries
        summarize=summarize,
        summarize_on=summarize_on,
        summary_methods=summary_methods,
        summary_opts=dict(
            num_sentences=xsum_sentences,
            sdg_targets=None,  # plug dict if you want SDG re-rank of summaries
            sdg_top_k=sdg_topk,
            cross_encoder_model=cross_encoder_model if cross_encoder_model else None,
            centroid_sim_threshold=0.55,
            centroid_top_n=5,
            centroid_store_vector=False,
        ),
        precomputed_embeddings=precomputed_embeddings,
    )
    return G, meta


# -------------------------- CLI entry --------------------------

def main():
    ap = argparse.ArgumentParser(description="Run SciCo graph with FAISS/MinHash-LSH shortlist and save viz")
    ap.add_argument("input", help="Input file (e.g., PDF) that SpanBERT quick pass can read")
    ap.add_argument("--max_sentences", type=int, default=40, help="Cap sentences from quick pass")
    ap.add_argument("--terms", nargs="*", default=[], help="Optional selected terms used as mentions")

    # Shortlist knobs
    ap.add_argument("--use_shortlist", action="store_true", help="Enable FAISS/LSH shortlist in the pipeline")
    ap.add_argument("--faiss_topk", type=int, default=32)
    ap.add_argument("--nprobe", type=int, default=8)
    ap.add_argument("--add_lsh", action="store_true", help="Add MinHash LSH on top of FAISS shortlist")
    ap.add_argument("--lsh_threshold", type=float, default=0.80)
    ap.add_argument("--minhash_k", type=int, default=5)
    ap.add_argument("--cheap_len_ratio", type=float, default=0.25)
    ap.add_argument("--cheap_jaccard", type=float, default=0.08)
    ap.add_argument("--use_coherence", action="store_true", help="Use SGNLP CoherenceMomentum in shortlist")
    ap.add_argument("--coherence_threshold", type=float, default=0.55)

    # Clustering / communities
    ap.add_argument("--clustering", choices=["auto","kmeans","torque","both","none"], default="auto")
    ap.add_argument("--kmeans_k", type=int, default=5)
    ap.add_argument("--community_on", choices=["all","corefer","parent_child"], default="all")
    ap.add_argument("--community_method", choices=["greedy","louvain","leiden","labelprop","none"], default="greedy")
    ap.add_argument("--prob_threshold", type=float, default=0.55)

    # Summaries
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--summarize_on", choices=["community","kmeans","torque"], default="community")
    ap.add_argument("--summary_methods", nargs="*", default=["centroid"], help="Any of: centroid xsum presumm")
    ap.add_argument("--xsum_sentences", type=int, default=1)
    ap.add_argument("--sdg_topk", type=int, default=3)
    ap.add_argument("--cross_encoder_model", default="cross-encoder/ms-marco-MiniLM-L6-v2")

    # Outputs
    ap.add_argument("--html", default="scico_graph.html", help="Output HTML path")
    ap.add_argument("--json", default="scico_meta.json", help="Optional meta JSON dump")
    ap.add_argument("--no_html", action="store_true", help="Skip HTML viz, only dump JSON meta")

    args = ap.parse_args()

    G, meta = build_scico_from_file(
        input_path=args.input,
        max_sentences=args.max_sentences,
        terms=args.terms,
        use_shortlist=args.use_shortlist,
        faiss_topk=args.faiss_topk,
        nprobe=args.nprobe,
        add_lsh=args.add_lsh,
        lsh_threshold=args.lsh_threshold,
        minhash_k=args.minhash_k,
        cheap_len_ratio=args.cheap_len_ratio,
        cheap_jaccard=args.cheap_jaccard,
        use_coherence=args.use_coherence,
        coherence_threshold=args.coherence_threshold,
        clustering=args.clustering,
        kmeans_k=args.kmeans_k,
        community_on=args.community_on,
        community_method=args.community_method,
        prob_threshold=args.prob_threshold,
        summarize=args.summarize,
        summarize_on=args.summarize_on,
        summary_methods=args.summary_methods,
        xsum_sentences=args.xsum_sentences,
        sdg_topk=args.sdg_topk,
        cross_encoder_model=args.cross_encoder_model,
    )

    # Save JSON
    if args.json:
        meta_out = clean_meta_for_json(meta)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, ensure_ascii=False, indent=2)
        print(f"[ok] saved meta to {args.json}")

        print(json.dumps({
            "pairs_scored": meta_out["pairs_scored"],
            "num_communities": len(set(meta_out["communities"].values())) if meta_out["communities"] else 0,
            "summary_groups": list(meta_out["summaries"].keys())[:10],
        }, ensure_ascii=False, indent=2))

    # Save HTML (unless skipped)
    if not args.no_html:
        html = pyvis_html(G, meta, height_px=800)
        Path(args.html).write_text(html, encoding="utf-8")
        print(f"[ok] graph: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
        print(f"[ok] html:  {args.html}")


if __name__ == "__main__":
    main()
