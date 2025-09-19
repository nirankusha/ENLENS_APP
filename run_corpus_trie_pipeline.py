# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# run_corpus_trie_pipeline.py
# One-shot end-to-end: PDFs -> production_output -> SQLite (documents/sentences/chains + trie) -> global query

import argparse, json, os, sys, re
from pathlib import Path

# Local modules
from helper import extract_text_from_pdf_robust, preprocess_pdf_text
from helper_addons import build_sentence_index, build_ngram_trie
from flexiconc_adapter import (
    open_db, 
    export_production_to_flexiconc, 
    upsert_doc_trie, 
    count_indices, 
    list_index_sizes
    )
from global_coref_helper import global_coref_query
 
def build_repetition_chains(full_text: str, *, min_len=4, min_df=2, max_chains=50):
    """
    Very light coref surrogate for testing:
      - find repeated tokens (length>=min_len) appearing >= min_df times
      - each unique token -> one chain with all its occurrences
    Mentions use absolute (start_char, end_char) in full_text.
    """
    # tokens to consider (letters/numbers/-/_), case-insensitive
    words = re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?", full_text.lower())
    freq = {}
    for w in words:
        if len(w) >= int(min_len):
            freq[w] = freq.get(w, 0) + 1
    # shortlist
    cands = [w for w, c in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])) if c >= int(min_df)]
    chains = []
    for cid, w in enumerate(cands[:max_chains]):
        # find all case-insensitive whole-word occurrences with char offsets
        ments = []
        for m in re.finditer(rf"(?i)\b{re.escape(w)}\b", full_text):
            s, e = m.span()
            ments.append({"start_char": s, "end_char": e, "text": full_text[s:e]})
        if len(ments) >= 2:
            chains.append({"chain_id": cid, "representative": w, "mentions": ments})
    return chains

def make_minimal_production_output(full_text: str):
    """
    Construct a minimal production_output:
      - full_text
      - sentence_analyses: (sentence_id, sentence_text, doc_start, doc_end)
      - coreference_analysis.chains: from repeated-token heuristic
    """
    sid2span, _tree = build_sentence_index(full_text)
    sents = []
    for sid in sorted(sid2span.keys()):
        st, en = sid2span[sid]
        sents.append({
            "sentence_id": int(sid),
            "sentence_text": full_text[st:en],
            "doc_start": int(st),
            "doc_end": int(en),
            "classification": {},
            "token_analysis": {},
            "span_analysis": []
        })
    chains = build_repetition_chains(full_text)
    return {
        "full_text": full_text,
        "sentence_analyses": sents,
        "coreference_analysis": {"chains": chains},
        "clusters": []
    }

def iter_pdfs(pdf_dir: Path, limit: int | None):
    files = []
    for ext in ("*.pdf", "*.PDF"):
        files.extend(sorted(pdf_dir.rglob(ext)))
    if limit:
        files = files[:int(limit)]
    return files

def main():
    ap = argparse.ArgumentParser(description="PDF dir -> SQLite (FlexiConc-style + global trie) -> query")
    ap.add_argument("--pdf-dir", required=True, help="Directory containing PDFs (recursively scanned)")
    ap.add_argument("--db-path", required=True, help="SQLite path to create (no preexisting DB required)")
    ap.add_argument("--out-json", help="(Optional) folder to save per-doc production_output.json")
    ap.add_argument("--limit", type=int, help="Max PDFs to process")
    ap.add_argument("--query", nargs="*", help="One or more span texts to query globally")
    ap.add_argument("--tau", type=float, default=0.18, help="Trie score threshold (IDF-Jaccard)")
    ap.add_argument("--topk", type=int, default=10, help="Max results to show")
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
        raw = extract_text_from_pdf_robust(pdf_path)
        text = preprocess_pdf_text(raw, max_length=None)

        # Minimal production output
        po = make_minimal_production_output(text)

        # Optionally save JSON
        if out_json_dir:
            jpath = out_json_dir / f"{doc_id}.json"
            jpath.write_text(json.dumps(po, ensure_ascii=False), encoding="utf-8")
            print(f"  ⬇️  Saved {jpath.name}")

        # Write structured rows (documents/sentences/chains)
        export_production_to_flexiconc(args.db_path, doc_id, po, uri=str(pdf_path))

        # Build trie and persist under indices(kind='trie')
        chains = (po.get("coreference_analysis") or {}).get("chains") or []
        if chains:
            root, idf, chain_grams = build_ngram_trie(chains, char_n=4, token_ns=(2,3))
            cx = open_db(args.db_path)
            try:
                upsert_doc_trie(cx, doc_id, root, idf, chain_grams)
                n = count_indices(cx, "trie")
                print(f"   indices(kind='trie'): {n} rows now")
                if n <= 3:
                    print("   sample:", list_index_sizes(cx, "trie", limit=3))
            finally:
                cx.close()
            print(f"  ✅ Indexed trie: chains={len(chains)}")
        else:
            print("  ⚠️  No chains built; skipping trie for this doc.")

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

