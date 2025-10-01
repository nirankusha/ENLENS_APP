# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# run_corpus_trie_pipeline.py
# One-shot end-to-end: PDFs -> production_output -> SQLite (documents/sentences/chains + trie) -> global query

import argparse, json, os, sys, re
from pathlib import Path

# Local modules
from helper import (
    extract_text_from_pdf_robust,
    preprocess_pdf_text,
    encode_mpnet,
    encode_sdg_hidden,
    encode_scico,
)
from helper_addons import build_sentence_index, build_ngram_trie
from flexiconc_adapter import (
    open_db, 
    count_indices, 
    list_index_sizes
    )
from global_coref_helper import global_coref_query
from bridge_runners import run_ingestion_quick

EMBEDDING_MODELS = {
    "mpnet": encode_mpnet,
    "sdg-bert": encode_sdg_hidden,
    "scico": encode_scico,
}
 

def iter_pdfs(pdf_dir: Path, limit: int | None):
    files = []
    for ext in ("*.pdf", "*.PDF"):
        files.extend(sorted(pdf_dir.rglob(ext)))
    if limit:
        files = files[:int(limit)]
    return files

def main():
    ap = argparse.ArgumentParser(
        description=(
            "PDF dir -> production pipeline (SpanBERT/KP) -> FlexiConc SQLite with trie/co-occ + query."
            "\nRequires ENLENS_SpanBert_corefree_prod.py (and optional ENLENS_KP_BERT_corefree_prod.py)."
        )
    )
    ap.add_argument("--pdf-dir", required=True, help="Directory containing PDFs (recursively scanned)")
    ap.add_argument("--db-path", required=True, help="SQLite path to create (no preexisting DB required)")
    ap.add_argument("--out-json", help="(Optional) folder to save per-doc production_output.json")
    ap.add_argument("--limit", type=int, help="Max PDFs to process")
    ap.add_argument("--query", nargs="*", help="One or more span texts to query globally")
    ap.add_argument("--tau", type=float, default=0.18, help="Trie score threshold (IDF-Jaccard)")
    ap.add_argument("--topk", type=int, default=10, help="Max results to show")
    ap.add_argument("--candidate-source", choices=["span", "kp"], default="span",
                    help="Which production pipeline to run (span=SpanBERT coref, kp=KPE coref)")
    ap.add_argument("--coref-device", default="auto",
                    help="Device for coreference backend (mirrors app config; e.g., 'cpu', 'cuda:0', 'auto')")
    ap.add_argument("--coref-shortlist-mode", choices=["off", "trie", "cooc", "both"], default="trie",
                    help="Shortlist strategy for coref candidate generation")
    ap.add_argument("--coref-shortlist-topk", type=int, default=50,
                    help="Top-K shortlist size for candidate generation")
    ap.add_argument("--cooc-mode", choices=["spacy", "hf"], default="spacy",
                    help="Co-occurrence builder (spaCy or Hugging Face tokenizer)")
    ap.add_argument("--cooc-hf-tokenizer",
                    help="Tokenizer name when --cooc-mode=hf (e.g., 'bert-base-uncased')")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        raise SystemExit(f"PDF dir not found: {pdf_dir}")

    # 1) Create/open DB (this will also ensure schema for documents/sentences/chains + indices)
    cx = open_db(args.db_path)
    cx.close()  # we will reopen per helper calls

    out_json_dir = Path(args.out_json) if args.out_json else None
    if out_json_dir:
        out_json_dir.mkdir(parents=True, exist_ok=True)

    # 2) Ingest each PDF -> production_output -> write to DB -> build trie
    pdfs = iter_pdfs(pdf_dir, args.limit)
    if not pdfs:
        print("No PDFs found.")
        return

    print(f"Found {len(pdfs)} PDFs.")
    for i, pdf_path in enumerate(pdfs, 1):
        doc_id = pdf_path.stem
        print(f"\n[{i}/{len(pdfs)}] Processing: {pdf_path.name}")

        # Extract + preprocess
        ingest_result = run_ingestion_quick(
            str(pdf_path),
            candidate_source=args.candidate_source,
            doc_id=doc_id,
            flexiconc_db_path=args.db_path,
            coref_device=args.coref_device,
            coref_shortlist_mode=args.coref_shortlist_mode,
            coref_shortlist_topk=args.coref_shortlist_topk,
            cooc_mode=args.cooc_mode,
            cooc_hf_tokenizer=args.cooc_hf_tokenizer,
        )

        if not ingest_result.get("ok"):
            print(f"  ❌ Ingestion failed: {ingest_result.get('error')}")
            continue
        # Minimal production output
        warnings = ingest_result.get("_warn") or []
        for w in warnings:
            print(f"  ⚠️  {w}")

        po = (ingest_result.get("production_output") or {}).copy()

        # Optionally save JSON
        if out_json_dir and po:
            jpath = out_json_dir / f"{doc_id}.json"
            jpath.write_text(json.dumps(po, ensure_ascii=False), encoding="utf-8")
            print(f"  ⬇️  Saved {jpath.name}")
            # Summarize trie row counts for visibility
        
        cx = open_db(args.db_path)
        try:
            n = count_indices(cx, "trie")
            print(f"   indices(kind='trie'): {n} rows now")
            if n <= 3:
                print("   sample:", list_index_sizes(cx, "trie", limit=3))
        finally:
            cx.close()
        print("  ✅ Ingestion complete via production pipeline")
    # 3) Queries (trie-only; co-occ optional later)
    if args.query:
        cx = open_db(args.db_path)
        try:
            for q in args.query:
                q = (q or "").strip()
                if not q:
                    continue
                hits = global_coref_query(q, cx, use_trie=True, use_cooc=False,
                                          topk=int(args.topk), tau_trie=float(args.tau))
                print("\n=== QUERY ===", q)
                if not hits:
                    print("(no hits)")
                for rank, h in enumerate(hits, 1):
                    print(f"{rank:02d}. doc={h.get('doc_id')} chain={h.get('chain_id')} "
                          f"score_trie={h.get('score_trie'):.3f}")
        finally:
            cx.close()
    else:
        print("\n[INFO] No --query provided. Add --query 'your phrase' to test retrieval.")

if __name__ == "__main__":
    main()

"""
Created on Mon Sep 15 11:12:40 2025

@author: niran
"""

