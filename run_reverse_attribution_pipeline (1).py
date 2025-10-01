# -*- coding: utf-8 -*-
"""
Reverse Attribution: Retrieval → Captum IG → (optional) Span refinement
======================================================================
[docstring trimmed for brevity]
"""
import os, re, json, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from helper import extract_text_from_pdf_robust, preprocess_pdf_text
from helper_addons import build_sentence_index

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from captum.attr import IntegratedGradients, DeepLift

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
    found: List[Path] = []
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


class Retriever:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs)
    def top_k(self, query: str, candidates: List[str], k: int = 3):
        q = self.encode([query])
        C = self.encode(candidates)
        sims = cosine_similarity(q, C)[0]
        idxs = np.argsort(-sims)[:k]
        return idxs.tolist(), sims[idxs].tolist()


class IGScorer:
    def __init__(self, model_name: str = "google/mt5-base", ig_steps: int = 32, use_deeplift: bool = False):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        self.ig_steps = ig_steps
        self.use_deeplift = use_deeplift

    def _forward_core(self, inputs_embeds, attention_mask, decoder_input_ids, decoder_attention_mask, labels):
        # IMPORTANT: disable cache; supply decoder ids + mask so shapes stay consistent
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
        # We return a scalar where higher = better support for the gold summary
        return -out.loss

    def token_attributions(self, src_text: str, tgt_text: str, enc_max: int = 768, lab_max: int = 256):
        enc = self.tok(
            src_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=enc_max,
        )
        lab = self.tok(
            tgt_text,
            return_tensors="pt",
            truncation=True,
            max_length=lab_max,
        )

        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        lab = {k: v.to(DEVICE) for k, v in lab.items()}

        # Prepare encoder inputs w/ grads
        for p in self.model.parameters():
            p.requires_grad_(False)
        inputs_embeds = self.model.get_input_embeddings()(enc["input_ids"])
        inputs_embeds.requires_grad_(True)

        # Prepare decoder inputs explicitly and the matching mask
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(lab["input_ids"])
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        decoder_attention_mask = (decoder_input_ids != pad_id).long().to(DEVICE)

        # Captum explainer
        forward_fn = lambda inp: self._forward_core(
            inp, enc["attention_mask"], decoder_input_ids, decoder_attention_mask, lab["input_ids"]
        )

        if self.use_deeplift:
            explainer = DeepLift(forward_fn)
            attrs = explainer.attribute(inputs=inputs_embeds, baselines=torch.zeros_like(inputs_embeds))
        else:
            explainer = IntegratedGradients(forward_fn)
            attrs = explainer.attribute(
                inputs=inputs_embeds,
                baselines=torch.zeros_like(inputs_embeds),
                n_steps=self.ig_steps,
            )

        token_attr = attrs.norm(dim=-1).squeeze(0)  # [T_enc]
        token_attr = token_attr / (token_attr.sum() + 1e-12)
        tokens = self.tok.convert_ids_to_tokens(enc["input_ids"].detach().cpu().tolist()[0])
        offsets = enc["offset_mapping"].detach().cpu().tolist()[0]
        return tokens, offsets, token_attr.detach().cpu().numpy()

    def sentence_score(self, sent: str, summary: str) -> float:
        _, _, a = self.token_attributions(sent, summary)
        return float(a.sum())


def spans_from_token_scores(text: str, offsets, scores, top_k: int = 1, smooth_window: int = 3,
                            min_span_len: int = 20, merge_gap: int = 15):
    import numpy as np
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
        if R - L + 1 >= min_span_len:
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


def parse_row_dois(row, col="all_dois"):
    import pandas as pd
    val = row.get(col, None)
    if pd.isna(val): return []
    if isinstance(val, (list, tuple)): return [str(x) for x in val]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("["):
            try:
                arr = eval(s)
                return [str(x) for x in arr]
            except Exception:
                return [t.strip() for t in s.split(",") if t.strip()]
        return [t.strip() for t in s.split(",") if t.strip()]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df_csv", required=True)
    ap.add_argument("--row_iloc", type=int, default=0)
    ap.add_argument("--src_dir", default="/content/drive/MyDrive/ENLENS/summ_pdfs")
    ap.add_argument("--out_prefix", default="/content/revattr_row")
    ap.add_argument("--summary_col", default="Green Hydrogen Value Chain Justification_Justification")
    ap.add_argument("--dois_col", default="all_dois")
    ap.add_argument("--retriever", default="all-mpnet-base-v2")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--ig_model", default="google/mt5-base")
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--use_deeplift", type=str, default="false")
    ap.add_argument("--refine_spans", type=str, default="true")
    args = ap.parse_args()

    use_deeplift = args.use_deeplift.lower().startswith("t")
    refine_spans = args.refine_spans.lower().startswith("t")

    df = pd.read_csv(args.df_csv)
    if not (0 <= args.row_iloc < len(df)):
        raise SystemExit(f"Row {args.row_iloc} out of bounds for df length {len(df)}")
    row = df.iloc[args.row_iloc]

    dois = parse_row_dois(row, args.dois_col)
    summary = str(row.get(args.summary_col, "")).strip()
    if not summary:
        raise SystemExit(f"No summary in '{args.summary_col}'")

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
        raise SystemExit("No documents parsed.")

    doc_ids = list(documents.keys())
    doc_sentences: Dict[str, List[str]] = {}
    for did in doc_ids:
        text = documents[did]
        sid2span = sentence_maps[did]["sid2span"]
        sents = [text[s:e].strip() for (s,e) in [sid2span[i] for i in range(len(sid2span))]]
        doc_sentences[did] = sents

    retr = Retriever(args.retriever)
    candidates_by_doc: Dict[str, List[Tuple[int, str, float]]] = {}
    for did in doc_ids:
        sents = doc_sentences[did]
        if not sents: continue
        k = min(args.top_k, len(sents))
        idxs, sims = retr.top_k(summary, sents, k=k)
        candidates_by_doc[did] = [(int(i), sents[int(i)], float(sims[j])) for j, i in enumerate(idxs)]

    ig = IGScorer(model_name=args.ig_model, ig_steps=args.ig_steps, use_deeplift=use_deeplift)

    selected_rows = []
    structured = []
    claim_id = 1

    for did in doc_ids:
        text = documents[did]
        sid2span = sentence_maps[did]["sid2span"]
        cand = candidates_by_doc.get(did, [])
        if not cand: continue

        ig_scored = []
        for sid, sent, sim in cand:
            try:
                score = ig.sentence_score(sent, summary)
            except Exception as e:
                print(f"⚠️ IG failed on a sentence in {did}: {e}")
                score = 0.0
            ig_scored.append((sid, sent, sim, score))

        ig_scored.sort(key=lambda x: x[3], reverse=True)
        if not ig_scored: continue
        sid, sent, sim, igscore = ig_scored[0]

        refined = []
        if refine_spans:
            tokens, offsets, scores = ig.token_attributions(sent, summary)
            refined = spans_from_token_scores(sent, offsets, scores, top_k=1)

        s_start, s_end = sid2span[sid]
        selected_rows.append({
            "claim_id": claim_id,
            "claim_text": summary,
            "source_doc": did,
            "sentence_idx": sid,
            "sentence": sent,
            "retrieval_sim": sim,
            "ig_score": igscore,
            "doc_char_start": s_start,
            "doc_char_end": s_end,
            "refined_span_start": (s_start + refined[0]["char_start"]) if refined else None,
            "refined_span_end": (s_start + refined[0]["char_end"]) if refined else None,
            "refined_span_text": refined[0]["snippet"] if refined else None,
        })

        structured.append({
            "claim_id": claim_id,
            "claim_text": summary,
            "evidence": [{
                "source_doc": did,
                "sentence_idx": sid,
                "sentence": sent,
                "retrieval_sim": sim,
                "ig_score": igscore,
                "doc_char_start": s_start,
                "doc_char_end": s_end,
                "refined_span": refined[0] if refined else None
            }]
        })

    out_json = f"{args.out_prefix}_result.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "total_claims": 1, "attributions": structured}, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved: {out_json}")

    out_csv = f"{args.out_prefix}_evidence.csv"
    pd.DataFrame(selected_rows).to_csv(out_csv, index=False)
    print(f"💾 Saved: {out_csv}")


if __name__ == "__main__":
    main()
