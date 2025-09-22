# -*- coding: utf-8 -*-
"""
Standalone tester for corpus-level global trie coref.
- Ingest: production_output.json -> per-doc trie in SQLite
- Query:  span_texts -> global_coref_query over superroot
"""

import argparse, json, os
from pathlib import Path

# DB + persistence helpers
from flexiconc_adapter import open_db, upsert_doc_trie, upsert_doc_cooc
# per-doc index builders
from helper_addons import build_ngram_trie, build_cooc_graph
# corpus-level coref query
from global_coref_helper import global_coref_query

def ingest_production_output(cx, prod_path: str, doc_id: str, *, also_cooc: bool = False):
    """Load a production_output JSON and persist trie (and optional cooc) for this doc."""
    with open(prod_path, "r", encoding="utf-8") as f:
        po = json.load(f)

    chains = (po.get("coreference_analysis") or {}).get("chains") or []
    if not chains:
        print(f"[WARN] No chains in {prod_path}; skipping trie for {doc_id}")
    else:
        trie_root, trie_idf, chain_grams = build_ngram_trie(chains, char_n=4, token_ns=(2,3))
        upsert_doc_trie(cx, doc_id, trie_root, trie_idf, chain_grams)
        print(f"[OK] Upserted trie for doc_id={doc_id}: chains={len(chains)}")

    if also_cooc:
        full_text = po.get("full_text") or ""
        if full_text.strip():
            vocab, rows, row_norms = build_cooc_graph(
                full_text,
                window=5,
                min_count=2,
                topk_neighbors=10,
                mode="spacy",
                )
            upsert_doc_cooc(cx, doc_id, vocab, rows, row_norms)
            print(f"[OK] Upserted cooc for doc_id={doc_id}: |V|={len(vocab)}")
        else:
            print(f"[WARN] No full_text in {prod_path}; skipping cooc for {doc_id}")

def run_queries(cx, queries, *, topk=10, tau_trie=0.18):
    """Query the global superroot using trie-only (fast path)."""
    for q in queries:
        q = q.strip()
        if not q:
            continue
        hits = global_coref_query(
            q, cx,
            use_trie=True, use_cooc=False,   # cooc off for pure-trie test
            topk=topk, tau_trie=float(tau_trie)
        )
        print("\n=== QUERY ===")
        print(q)
        if not hits:
            print("(no hits)")
            continue
        for i, h in enumerate(hits, 1):
            # h has: doc_id, chain_id, score_trie, score, why
            print(f"{i:02d}. doc={h.get('doc_id')}  chain={h.get('chain_id')}  "
                  f"score_trie={h.get('score_trie'):.3f}  why={h.get('why','trie')}")

def main():
    ap = argparse.ArgumentParser(description="Test global trie coref over FlexiConc DB")
    ap.add_argument("--db", required=True, help="Path to SQLite DB (will be created if missing)")
    ap.add_argument("--ingest", nargs="*", help="One or more production_output.json files to index")
    ap.add_argument("--doc-ids", nargs="*", help="Optional doc_ids matching --ingest (defaults to file stem)")
    ap.add_argument("--also-cooc", action="store_true", help="Also build co-occ matrix (optional)")
    ap.add_argument("--query", nargs="*", help="One or more span texts to query")
    ap.add_argument("--tau", type=float, default=0.18, help="Trie score threshold (IDF-Jaccard)")
    ap.add_argument("--topk", type=int, default=10, help="Max results to show")
    args = ap.parse_args()

    cx = open_db(args.db)

    # Optional ingestion step
    if args.ingest:
        doc_ids = args.doc_ids or []
        if doc_ids and len(doc_ids) != len(args.ingest):
            raise SystemExit("--doc-ids must match the number of --ingest files")
        for i, p in enumerate(args.ingest):
            did = (doc_ids[i] if doc_ids else Path(p).stem)
            ingest_production_output(cx, p, did, also_cooc=args.also_cooc)

    # Queries
    if args.query:
        run_queries(cx, args.query, topk=args.topk, tau_trie=args.tau)
    else:
        print("[INFO] No --query provided. Use --query 'your span here' ...")

if __name__ == "__main__":
    main()

"""
Created on Fri Sep 12 17:04:11 2025

@author: niran
"""

