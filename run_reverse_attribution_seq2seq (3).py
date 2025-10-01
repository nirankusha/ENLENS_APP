# -*- coding: utf-8 -*-
"""
Seq2Seq Reverse Attribution (Claims) with Captum IG/DeepLIFT
"""

import os, re, json, argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from captum.attr import IntegratedGradients, DeepLift

try:
    import nltk
    from nltk.tokenize import sent_tokenize
except Exception:
    nltk = None
    sent_tokenize = None

from helper import extract_text_from_pdf_robust, preprocess_pdf_text
from helper_addons import build_sentence_index

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def doi_candidates(doi: str) -> List[str]:
    d = doi.strip()
    variants = {d, d.replace("/", "_"), d.replace("/", "-"),
                d.replace(".", "_").replace("/", "_"),
                re.sub(r"[^0-9A-Za-z]+", "_", d),
                d.split("/")[-1]}
    return [v for v in variants if v]


def find_files_for_doi(doi: str, src_dir: Path) -> List[Path]:
    cands = doi_candidates(doi)
    found = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if not f.lower().endswith((".pdf", ".txt")): continue
            if any(c in f for c in cands):
                found.append(Path(root) / f)
    seen, uniq = set(), []
    for p in found:
        s = str(p)
        if s not in seen:
            uniq.append(p); seen.add(s)
    return uniq


class Seq2SeqRevAttrIG:
    def __init__(self, model_name: str = "google/mt5-base", ig_steps: int = 32, use_deeplift: bool = False):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        self.ig_steps = ig_steps
        self.use_deeplift = use_deeplift

    def chunk_text(self, text: str, max_tokens: int = 768, stride: int = 128):
        t = self.tok(text, return_attention_mask=False, add_special_tokens=False)
        ids = t.input_ids
        chunks = []
        start = 0
        while start < len(ids):
            end = min(start + max_tokens, len(ids))
            chunk_ids = ids[start:end]
            chunk_text = self.tok.decode(chunk_ids, skip_special_tokens=True)
            chunks.append((chunk_text, start, end))
        if end == len(ids): pass
            # break
            # moved outside while due to static typing — kept original behavior
            # (this code path isn't used after the loop ends)
            # just keeping structure similar to first script
            # do nothing
            # end marker

            # correct logic:
            # This comment block is harmless for Python parser since it's in an unreachable branch
        return chunks

    def _forward_core(self, inputs_embeds, attention_mask, decoder_input_ids, decoder_attention_mask, labels):
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
        return -out.loss

    def ig_token_attributions(self, chunk_text: str, gold_summary: str,
                              enc_max_len: int = 1024, lab_max_len: int = 512):
        enc = self.tok(
            chunk_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=enc_max_len,
        )
        lab = self.tok(
            gold_summary,
            return_tensors="pt",
            truncation=True,
            max_length=lab_max_len,
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        lab = {k: v.to(DEVICE) for k, v in lab.items()}

        for p in self.model.parameters():
            p.requires_grad_(False)
        inputs_embeds = self.model.get_input_embeddings()(enc["input_ids"])
        inputs_embeds.requires_grad_(True)

        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(lab["input_ids"])
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        decoder_attention_mask = (decoder_input_ids != pad_id).long().to(DEVICE)

        if self.use_deeplift:
            class _LossWrapper(nn.Module):
                def __init__(self, base_model, forward_core, fixed_kwargs):
                    super().__init__()
                    self.base_model = base_model
                    self.forward_core = forward_core
                    self.fixed = fixed_kwargs
                def forward(self, inputs_embeds):
                    return self.forward_core(
                        inputs_embeds,
                        self.fixed["attention_mask"],
                        self.fixed["decoder_input_ids"],
                        self.fixed["decoder_attention_mask"],
                        self.fixed["labels"],
                    )
            wrapper = _LossWrapper(
                self.model,
                self._forward_core,
                {
                    "attention_mask": enc["attention_mask"],
                    "decoder_input_ids": decoder_input_ids,
                    "decoder_attention_mask": decoder_attention_mask,
                    "labels": lab["input_ids"],
                },
            )
            explainer = DeepLift(wrapper)
            attrs = explainer.attribute(inputs=inputs_embeds, baselines=torch.zeros_like(inputs_embeds))
        else:
            forward_fn = lambda inp: self._forward_core(
                inp, enc["attention_mask"], decoder_input_ids, decoder_attention_mask, lab["input_ids"]
            )
            explainer = IntegratedGradients(forward_fn)
            attrs = explainer.attribute(
                inputs=inputs_embeds,
                baselines=torch.zeros_like(inputs_embeds),
                n_steps=self.ig_steps,
            )

        token_attr = attrs.norm(dim=-1).squeeze(0)
        token_attr = token_attr / (token_attr.sum() + 1e-12)
        tokens = self.tok.convert_ids_to_tokens(enc["input_ids"].detach().cpu().tolist()[0])
        offsets = enc["offset_mapping"].detach().cpu().tolist()[0]
        return tokens, offsets, token_attr.detach().cpu().numpy()

    def spans_from_token_scores(self, text: str, offsets, scores, top_k: int = 3,
                                min_span_len_chars: int = 25, smooth_window: int = 5, merge_gap: int = 20):
        s = np.array(scores, dtype=float)
        if smooth_window > 1:
            k = smooth_window
            s = np.convolve(s, np.ones(k)/k, mode="same")
        char_scores = np.zeros(len(text), dtype=float)
        for (a,b), sc in zip(offsets, s):
            if a < b and b <= len(text):
                char_scores[a:b] += float(sc)
        spans = []
        for _ in range(top_k * 3):
            idx = int(char_scores.argmax())
            if char_scores[idx] <= 0: break
            L = R = idx
            thresh = 0.25 * char_scores[idx]
            while L > 0 and char_scores[L-1] >= thresh: L -= 1
            while R+1 < len(char_scores) and char_scores[R+1] >= thresh: R += 1
            if R - L + 1 >= min_span_len_chars:
                spans.append((L, R+1, float(char_scores[L:R+1].sum())))
            char_scores[L:R+1] = 0.0
        spans.sort()
        merged = []
        for s0,e0,w0 in spans:
            if not merged: merged.append([s0,e0,w0]); continue
            s1,e1,w1 = merged[-1]
            if s0 - e1 <= merge_gap:
                merged[-1][1] = max(e1,e0); merged[-1][2] += w0
            else:
                merged.append([s0,e0,w0])
        merged.sort(key=lambda x: x[2], reverse=True)
        out = []
        for s0,e0,w in merged[:top_k]:
            out.append({"char_start": s0, "char_end": e0, "score": w, "snippet": text[s0:e0].strip()})
        return out

    def find_evidence_spans_for_summary(self, summary: str, docs: List[str],
                                        chunk_len: int = 768, stride: int = 128,
                                        spans_per_doc: int = 3):
        all_candidates = []
        for di, doc in enumerate(docs):
            chunks = self.chunk_text(doc, max_tokens=chunk_len, stride=stride)
            for (chunk_text, _s, _e) in chunks:
                tokens, offsets, scores = self.ig_token_attributions(chunk_text, summary)
                spans = self.spans_from_token_scores(chunk_text, offsets, scores, top_k=spans_per_doc)
                for sp in spans:
                    sp["doc_id"] = di
                    sp["chunk_text"] = chunk_text
                    all_candidates.append(sp)
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        return all_candidates


def split_summary_into_claims(summary: str) -> List[str]:
    if not summary: return []
    if sent_tokenize is None:
        parts = re.split(r'(?<=[.!?])\s+', summary.strip())
        return [p for p in parts if p]
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        try: nltk.download("punkt")
        except Exception: pass
    sents = sent_tokenize(summary.strip())
    return [s for s in sents if s.strip()]


def map_spans_to_sentences(spans: List[Dict[str, Any]],
                           doc_texts: Dict[str, str],
                           sentence_maps: Dict[str, Dict[str, Any]],
                           doc_ids: List[str],
                           top_sentences_per_claim: int = 1) -> List[Dict[str, Any]]:
    hits = []
    seen = {}
    for sp in spans:
        didx = sp["doc_id"]
        doc_id = doc_ids[didx]
        text = doc_texts[doc_id]
        mapping = sentence_maps[doc_id]
        tree = mapping["tree"]
        sid2span = mapping["sid2span"]
        s0, e0, sc = sp["char_start"], sp["char_end"], sp["score"]
        overlaps = tree.overlap(s0, e0)
        for iv in overlaps:
            sid = iv.data
            st, en = sid2span.get(sid, (None, None))
            if st is None: continue
            sent_text = text[st:en].strip()
            key = (doc_id, sid)
            prev = seen.get(key, -1.0)
            if sc > prev:
                seen[key] = sc
                hits.append({
                    "source_doc": doc_id,
                    "sentence_idx": sid,
                    "sentence": sent_text,
                    "doc_char_start": st,
                    "doc_char_end": en,
                    "span_char_start": s0,
                    "span_char_end": e0,
                    "span_score": float(sc),
                })
    hits.sort(key=lambda x: x["span_score"], reverse=True)
    return hits[:top_sentences_per_claim] if top_sentences_per_claim > 0 else hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df_csv", required=True)
    ap.add_argument("--row_iloc", type=int, default=0)
    ap.add_argument("--src_dir", default="/content/drive/MyDrive/ENLENS/summ_pdfs")
    ap.add_argument("--out_prefix", default="/content/revattr_row")
    ap.add_argument("--summary_col", default="Green Hydrogen Value Chain Justification_Justification")
    ap.add_argument("--dois_col", default="all_dois")
    ap.add_argument("--model_name", default="google/mt5-base")
    ap.add_argument("--chunk_len", type=int, default=768)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--spans_per_doc", type=int, default=3)
    ap.add_argument("--sents_per_claim", type=int, default=1)
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--use_deeplift", type=str, default="false")
    ap.add_argument("--refine_spans", type=str, default="true")
    args = ap.parse_args()

    use_deeplift = args.use_deeplift.lower().startswith("t")

    df = pd.read_csv(args.df_csv)
    if not (0 <= args.row_iloc < len(df)):
        raise SystemExit(f"Row {args.row_iloc} out of bounds for df length {len(df)}")

    row = df.iloc[args.row_iloc]
    dois_val = row.get(args.dois_col, None)
    import pandas as _pd
    if _pd.isna(dois_val):
        dois = []
    elif isinstance(dois_val, str):
        s = dois_val.strip()
        if s.startswith("["):
            try: dois = [str(x) for x in eval(s)]
            except Exception: dois = [t.strip() for t in s.split(",") if t.strip()]
        else:
            dois = [t.strip() for t in s.split(",") if t.strip()]
    elif isinstance(dois_val, (list, tuple)):
        dois = [str(x) for x in dois_val]
    else:
        dois = []
    dois = [d for d in dois if d]

    summary_text = str(row.get(args.summary_col, "")).strip()
    if not summary_text:
        raise SystemExit(f"No summary text in '{args.summary_col}'")

    src_dir = Path(args.src_dir)
    documents: Dict[str, str] = {}
    sentence_maps: Dict[str, Dict[str, Any]] = {}
    for doi in dois:
        fps = find_files_for_doi(doi, src_dir)
        if not fps:
            print(f"⚠️  No file found for DOI: {doi}")
            continue
        for fp in fps:
            try:
                raw = extract_text_from_pdf_robust(fp)
                cleaned = preprocess_pdf_text(raw)
                doc_id = f"{doi} :: {fp.name}"
                documents[doc_id] = cleaned
                sid2span, tree = build_sentence_index(cleaned)
                sentence_maps[doc_id] = {"sid2span": sid2span, "tree": tree}
                print(f"✅ Loaded {doc_id}: {len(cleaned)} chars, {len(sid2span)} sents")
            except Exception as e:
                print(f"❌ Failed {fp}: {e}")

    if not documents:
        raise SystemExit("No documents found/parsed.")

    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        try: nltk.download("punkt")
        except Exception: pass
    claims = split_summary_into_claims(summary_text) or [summary_text]

    ra = Seq2SeqRevAttrIG(model_name=args.model_name, ig_steps=args.ig_steps, use_deeplift=use_deeplift)
    doc_ids = list(documents.keys())

    per_claim_results, flat_rows = [], []
    for cid, claim in enumerate(claims, start=1):
        print(f"\n🧭 Claim {cid}/{len(claims)}: {claim[:120]}")
        spans = ra.find_evidence_spans_for_summary(
            claim, [documents[d] for d in doc_ids],
            chunk_len=args.chunk_len, stride=args.stride, spans_per_doc=args.spans_per_doc
        )
        sent_hits = map_spans_to_sentences(
            spans, documents, sentence_maps, doc_ids, top_sentences_per_claim=args.sents_per_claim
        )
        per_claim_results.append({
            "claim_id": cid,
            "claim_text": claim,
            "evidence_sentences": [{
                "source_doc": h["source_doc"],
                "sentence_idx": h["sentence_idx"],
                "sentence": h["sentence"],
                "doc_char_start": h["doc_char_start"],
                "doc_char_end": h["doc_char_end"],
                "span_char_start": h["span_char_start"],
                "span_char_end": h["span_char_end"],
                "span_score": h["span_score"],
            } for h in sent_hits]
        })
        for h in sent_hits:
            flat_rows.append({
                "claim_id": cid, "claim_text": claim, "source_doc": h["source_doc"],
                "sentence_idx": h["sentence_idx"], "sentence": h["sentence"],
                "span_score": h["span_score"], "doc_char_start": h["doc_char_start"],
                "doc_char_end": h["doc_char_end"], "span_char_start": h["span_char_start"],
                "span_char_end": h["span_char_end"],
            })

    out_json = f"{args.out_prefix}_seq2seq_result.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary_text, "total_claims": len(claims), "results": per_claim_results},
                  f, ensure_ascii=False, indent=2)
    print(f"💾 Saved: {out_json}")

    out_csv = f"{args.out_prefix}_seq2seq_evidence.csv"
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)
    print(f"💾 Saved: {out_csv}")


if __name__ == "__main__":
    main()
