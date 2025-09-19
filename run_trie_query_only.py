# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# run_trie_query_only.py

import argparse
from global_coref_helper import global_coref_query
from flexiconc_adapter import open_db, count_indices, list_index_sizes
    
def main():
    ap = argparse.ArgumentParser(description="Run global trie queries on an existing DB")
    ap.add_argument("--db-path", required=True, help="SQLite DB with trie indices")
    ap.add_argument("--query", nargs="+", required=True, help="One or more span texts to query")
    ap.add_argument("--tau", type=float, default=0.18, help="Trie score threshold (IDF-Jaccard)")
    ap.add_argument("--topk", type=int, default=10, help="Max results to show")
    args = ap.parse_args()


    cx = open_db(args.db_path)
    n = count_indices(cx, "trie")
    print(f"🔧 indices(kind='trie') rows: {n}")
    if n == 0:
        raise SystemExit("❌ No trie indices found. Ingest first.")
        print("   sample payload sizes:", list_index_sizes(cx, "trie", limit=3))
    try:
        for q in args.query:
            q = (q or "").strip()
            if not q:
                continue
            hits = global_coref_query(
                q, cx,
                use_trie=True, use_cooc=False,
                topk=int(args.topk),
                tau_trie=float(args.tau)
            )
            print("\n=== QUERY ===", q)
            if not hits:
                print("(no hits)")
            for rank, h in enumerate(hits, 1):
                print(f"{rank:02d}. doc={h.get('doc_id')} "
                      f"chain={h.get('chain_id')} "
                      f"score_trie={h.get('score_trie'):.3f}")
    finally:
        cx.close()

if __name__ == "__main__":
    main()

"""
Created on Mon Sep 15 11:35:53 2025

@author: niran
"""

