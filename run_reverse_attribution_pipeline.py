
# -*- coding: utf-8 -*-
"""
Run Reverse Attribution for a row in your dataframe
===================================================

What it does (per your spec):
1) Pick a .iloc row from a dataframe (CSV or an in-memory df you save to CSV).
2) Read that row's `all_dois` (and fallbacks).
3) Locate corresponding source files under a given folder (/content/drive/MyDrive/ENLENS/summ_pdfs).
4) Extract text from each PDF/TXT with helper.extract_text_from_pdf_robust -> preprocess -> sentence filtering.
5) Build a sentence index (IntervalTree) for each doc (helper_addons.build_sentence_index) for alignment.
6) Run reverse attribution from the row's human summary
   (`Green Hydrogen Value Chain Justification_Justification`) to the sentences from all located docs.
7) Produce *only the selected sentences* as evidence per summary sentence (claim).
8) Save tidy CSV + JSON with everything you need for downstream visualization.

Usage (in Colab or locally):
----------------------------
!pip install sentence-transformers scikit-learn nltk pandas
# Ensure helper.py, helper_addons.py, reverse_attribution.py are importable (same folder or sys.path).
python /mnt/data/run_reverse_attribution_pipeline.py \
  --df_csv /content/df_processed.csv \
  --row_iloc 0 \
  --src_dir /content/drive/MyDrive/ENLENS/summ_pdfs \
  --out_prefix /content/revattr_row0 \
  --min_sim 0.35 \
  --top_k 1

Outputs:
  <out_prefix>_evidence.csv  — flattened evidence table (one row per selected sentence)
  <out_prefix>_result.json   — full structured result
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd

# --- Local project imports (these must be available) ---
#   You uploaded these as /mnt/data/*.py; make sure your runtime sees them on sys.path.
try:
    from helper import (
        extract_text_from_pdf_robust,
        preprocess_pdf_text,
        extract_and_filter_sentences,
    )
    from helper_addons import build_sentence_index
    from reverse_attribution import StandaloneReverseAttributor
except Exception as e:
    raise SystemExit(f"❌ Could not import required modules (helper/helper_addons/reverse_attribution): {e}")


# ------------------------
# DOI → filename helpers
# ------------------------
def doi_candidates(doi: str) -> List[str]:
    """
    Return a list of plausible filename substrings derived from a DOI.
    We'll use them to search with glob/matching in the src_dir.
    """
    d = doi.strip()
    variants = set()
    variants.add(d)                                   # raw DOI
    variants.add(d.replace("/", "_"))                 # slashes -> underscores
    variants.add(d.replace("/", "-"))
    variants.add(d.replace(".", "_").replace("/", "_"))
    variants.add(re.sub(r"[^0-9A-Za-z]+", "_", d))    # keep only word chars
    variants.add(d.split("/")[-1])                    # suffix only
    return [v for v in variants if v]


def find_files_for_doi(doi: str, src_dir: Path) -> List[Path]:
    """
    Search src_dir recursively for files whose name contains any of our DOI variants.
    Accept .pdf and .txt.
    """
    cands = doi_candidates(doi)
    found: List[Path] = []
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if not f.lower().endswith((".pdf", ".txt")):
                continue
            name = f
            for c in cands:
                if c in name:
                    found.append(Path(root) / f)
                    break
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for p in found:
        if str(p) not in seen:
            uniq.append(p)
            seen.add(str(p))
    return uniq


# -------------------------------
# Build documents & sentence maps
# -------------------------------
def load_docs_from_dois(dois: List[str], src_dir: Path) -> Tuple[Dict[str, str], Dict[str, Dict]]:
    """
    Returns:
      documents: {doc_id: full_text}
      sentence_maps: {doc_id: {"sid2span": {sid:(s,e)}, "tree": IntervalTree}}
    """
    documents: Dict[str, str] = {}
    sentence_maps: Dict[str, Dict[str, Any]] = {}

    for doi in dois:
        files = find_files_for_doi(doi, src_dir)
        if not files:
            print(f"⚠️  No file found for DOI: {doi}")
            continue

        for fp in files:
            try:
                raw = extract_text_from_pdf_robust(fp)
                cleaned = preprocess_pdf_text(raw)
                # Keep full text; StandaloneReverseAttributor will split to sentences internally
                doc_id = f"{doi} :: {fp.name}"
                documents[doc_id] = cleaned

                # Build sentence index for alignment downstream
                sid2span, tree = build_sentence_index(cleaned)
                sentence_maps[doc_id] = {"sid2span": sid2span, "tree": tree}
                print(f"✅ Loaded {doc_id}: {len(cleaned)} chars, {len(sid2span)} sents")
            except Exception as e:
                print(f"❌ Failed to read {fp}: {e}")
    return documents, sentence_maps


# -------------------------------
# Flatten evidence to a tidy table
# -------------------------------
def flatten_evidence(overall_result, sentence_maps: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Produce one row per selected *sentence* of evidence.
    Columns:
      claim_id, claim_text, source_doc, sentence_idx, sentence, similarity, strength,
      char_start, char_end
    """
    rows = []
    for attr in overall_result.attributions:
        c_id = attr.claim_id
        c_txt = attr.claim_text
        for ev in attr.evidence:
            doc_id = ev.get("source_doc")
            sent = ev.get("sentence", "")
            sim = float(ev.get("similarity", 0.0))
            strength = ev.get("strength", "")
            sid = int(ev.get("sentence_idx", -1)) if "sentence_idx" in ev else -1

            # Try to recover sentence char span via sentence_maps[doc_id]
            s_start = s_end = None
            if doc_id in sentence_maps and sid >= 0:
                s_start, s_end = sentence_maps[doc_id]["sid2span"].get(sid, (None, None))

            rows.append({
                "claim_id": c_id,
                "claim_text": c_txt,
                "source_doc": doc_id,
                "sentence_idx": sid,
                "sentence": sent,
                "similarity": sim,
                "strength": strength,
                "char_start": s_start,
                "char_end": s_end,
            })
    return pd.DataFrame(rows)


# -------------------------------
# Main: argument parsing & run
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df_csv", required=True, help="Path to the dataframe CSV (must include 'all_dois' and the summary column).")
    ap.add_argument("--row_iloc", type=int, default=0, help="Row index (iloc) to process.")
    ap.add_argument("--src_dir", default="/content/drive/MyDrive/ENLENS/summ_pdfs", help="Folder with your source PDFs/TXTs.")
    ap.add_argument("--out_prefix", default="/content/revattr_row", help="Prefix for output files (CSV/JSON).")
    ap.add_argument("--summary_col", default="Green Hydrogen Value Chain Justification_Justification",
                    help="Column that holds the human summary text.")
    ap.add_argument("--dois_col", default="all_dois", help="Column that holds the list of DOIs.")
    ap.add_argument("--min_sim", type=float, default=0.35, help="Min cosine similarity to keep an evidence sentence.")
    ap.add_argument("--top_k", type=int, default=1, help="How many sentences per claim to keep.")
    ap.add_argument("--claims", choices=["sentences", "semantic_chunks", "single"], default="sentences",
                    help="How to split the summary into claims.")
    args = ap.parse_args()

    df = pd.read_csv(args.df_csv)
    if args.row_iloc < 0 or args.row_iloc >= len(df):
        raise SystemExit(f"Row iloc {args.row_iloc} is out of bounds for df with length {len(df)}")

    row = df.iloc[args.row_iloc]
    # Get DOIs list (robustly)
    dois_val = row.get(args.dois_col, None)
    if pd.isna(dois_val):
        dois = []
    elif isinstance(dois_val, str):
        # Could be "['doi1','doi2']" or comma-separated
        if dois_val.strip().startswith("["):
            try:
                dois = eval(dois_val)
            except Exception:
                dois = [s.strip() for s in dois_val.split(",") if s.strip()]
        else:
            dois = [s.strip() for s in dois_val.split(",") if s.strip()]
    elif isinstance(dois_val, (list, tuple)):
        dois = list(dois_val)
    else:
        dois = []

    dois = [d for d in dois if isinstance(d, str) and d.strip()]
    print(f"🔎 Row {args.row_iloc}: {len(dois)} DOIs from '{args.dois_col}'")

    # Load documents
    src_dir = Path(args.src_dir)
    documents, sentence_maps = load_docs_from_dois(dois, src_dir)
    if not documents:
        raise SystemExit("No documents found for the provided DOIs. Check your src_dir and filenames.")

    # Pull the human summary
    summary_text = str(row.get(args.summary_col, "")).strip()
    if not summary_text:
        raise SystemExit(f"No summary text found in column '{args.summary_col}' for iloc={args.row_iloc}")

    # Run reverse attribution
    attributor = StandaloneReverseAttributor()
    attributor.add_documents(documents)
    result = attributor.attribute_summary(
        summary=summary_text,
        top_k=args.top_k,
        min_similarity=args.min_sim,
        claim_extraction_method=args.claims,
    )

    # Attach sentence indices to evidence (by matching exact sentence text where possible)
    # Walk each doc_id's sentences to map text -> idx
    for doc_id, sents in attributor.corpus.items():
        text2idx = {sents[i]: i for i in range(len(sents))}
        for attr in result.attributions:
            for ev in attr.evidence:
                if ev.get("source_doc") == doc_id and "sentence_idx" not in ev:
                    idx = text2idx.get(ev.get("sentence", ""), -1)
                    ev["sentence_idx"] = idx

    # Save JSON
    out_json = f"{args.out_prefix}_result.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "summary": result.original_summary,
            "total_claims": result.total_claims,
            "stats": result.stats,
            "attributions": [
                {
                    "claim_id": a.claim_id,
                    "claim_text": a.claim_text,
                    "evidence": a.evidence,
                    "evidence_count": a.evidence_count,
                } for a in result.attributions
            ],
        }, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved: {out_json}")

    # Save flattened CSV (only selected sentences)
    df_ev = flatten_evidence(result, sentence_maps)
    out_csv = f"{args.out_prefix}_evidence.csv"
    df_ev.to_csv(out_csv, index=False)
    print(f"💾 Saved: {out_csv}")

    # Also print a compact preview
    print("\n=== Evidence preview (top 10) ===")
    with pd.option_context("display.max_colwidth", 120):
        print(df_ev.sort_values(["claim_id", "similarity"], ascending=[True, False]).head(10))

if __name__ == "__main__":
    main()
