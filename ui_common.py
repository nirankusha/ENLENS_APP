"""
ui_common.py - shared UI helpers (standalone)

"production_output" schema (normalized):
production_output = {
  "full_text": str,
  "sentence_analyses": [
     {"sentence_id": int,
      "sentence_text": str,
      "doc_start": int,
      "doc_end": int,
      "classification": {"label": int|str, "confidence": float, "consensus": str, "code": str|None},
      "token_analysis": {"tokens": [{"token": str, "importance": float, "start_char": int, "end_char": int}],
                          "max_importance": float, "num_tokens": int},
      "span_analysis": [
         {"rank": int,
          "original_span": {"text": str, "start_char": int, "end_char": int, "importance": float},
          "expanded_phrase": str,
          "coords": [int, int],
          "coreference_analysis": {"chain_found": bool, "chain_id"?: int, "representative"?: str}}
      ]}
  ],
  "coreference_analysis": {"num_chains": int, "chains": [...]},
  "cluster_analysis": {...}
}
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Literal, Optional
import html

# -------- Dropdown labels --------
def _count_spans_kp_coref(sa_item):
    # spans
    spans = sa_item.get("span_analysis") or []
    sp_total = len(spans)
    sp_coref = 0
    for s in spans:
        ca = (s.get("coreference_analysis") or {})
        if ca.get("chain_found") or ca.get("chain_id") is not None:
            sp_coref += 1
    # keyphrases (if present)
    kps = sa_item.get("keyphrase_analysis") or []
    kp_total, kp_coref = 0, 0
    if isinstance(kps, list):
        kp_total = len(kps)
        for k in kps:
            ca = (k.get("coreference_analysis") or {})
            if ca.get("chain_found") or ca.get("chain_id") is not None:
                kp_coref += 1
    return (sp_total, sp_coref, kp_total, kp_coref)

def format_sentence_option(sa_item, source: Literal["span","kp","auto"] = "auto") -> str:
    i = int(sa_item.get("sentence_id", 0))
    sent = (sa_item.get("sentence_text") or "").replace("\n", " ")
    snippet = (sent[:60] + ("---" if len(sent) > 60 else ""))
    cls = sa_item.get("classification", {})
    consensus = cls.get("consensus", "?")
    label = cls.get("label", "?")
    conf = cls.get("confidence", None)
    conf_txt = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
    sp_total, sp_coref, kp_total, kp_coref = _count_spans_kp_coref(sa_item)
    # Decide badge code
    if source == "auto":
        source = "span" if sp_total else "kp"
    code = "SP" if source == "span" else "KP"
    # Rich label: ID: label | conf | consensus | SP=a(b) | KP=c(d) | snippet
    return (f"{i}: {label} | conf={conf_txt} | {consensus} | "
            f"SP={sp_total}({sp_coref}) | KP={kp_total}({kp_coref}) | {snippet}")

def build_sentence_options(production_output: Dict[str, Any],
                           source: Literal["span","kp","auto"] = "auto") -> Tuple[List[str], List[int]]:
    items = production_output.get("sentence_analyses") or []
    labels = [format_sentence_option(sa, source=source) for sa in items]
    indices = [int(sa.get("sentence_id", idx)) for idx, sa in enumerate(items)]
    return labels, indices

# -------- HTML overlay --------
# put near the top (after imports)
# -*- coding: utf-8 -*-
import html
from typing import Dict, Any

CSS_BASE = """
<style>
.sbar {padding:6px 10px;border:1px solid #e5e7eb;border-radius:10px;background:#f7fafc;margin:6px 0}
.stitle {font-weight:600}
.smeta {color:#555;font-size:12px;margin-top:4px}
.sentbox {
  padding:10px 12px;border:1px solid #e5e7eb;border-radius:10px;background:#fff;
  max-width:100%;
  word-break: break-word;
  overflow-wrap: anywhere;
  white-space: normal;
  line-height: 1.9;
}
.overlay-scroll { max-height: 38vh; overflow:auto; }
.tk {
  padding:1px 2px;border-radius:4px;
  display:inline;
  white-space: pre-wrap;
  hyphens: auto;
}
.span-box {border:1px dashed #aaa;border-radius:6px;padding:0 2px}
.badge {display:inline-block;padding:2px 6px;border-radius:999px;background:#eef2ff;border:1px solid #d0d7ff;font-size:12px}
</style>
"""

def _normalize_importances(tokens: List[Dict[str, Any]]) -> List[float]:
    vals = [float(t.get("importance", 0.0)) for t in tokens]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    rng = hi - lo if hi != lo else 1.0
    return [(v - lo) / rng for v in vals]

def render_sentence_overlay(
    production_output: Dict[str, Any],
    sentence_id: int,
    highlight_coref: bool = True,
    box_spans: bool = True
) -> str:
    items = production_output.get("sentence_analyses") or []
    lookup = {int(sa.get("sentence_id", i)): sa for i, sa in enumerate(items)}
    sa = lookup.get(int(sentence_id))
    if not sa:
        return "<div>Sentence not found.</div>"

    sent: str = sa.get("sentence_text", "") or ""
    cls = sa.get("classification", {}) or {}
    cons = html.escape(str(cls.get("consensus", "?")))
    lab  = html.escape(str(cls.get("label", "?")))
    conf = cls.get("confidence", None)
    conf_txt = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"

    # tokens & normalization (if you have a helper)
    toks = (sa.get("token_analysis") or {}).get("tokens") or []
    norm = _normalize_importances(toks) if "_normalize_importances" in globals() else None

    base = int(sa.get("doc_start", 0))
    n = len(sent)

    # per-position HTML inserts we will stitch between characters
    inserts: List[str] = ["" for _ in range(n + 1)]

    # 1) span boxes (absolute -> sentence-local)
    if box_spans:
        for sp in (sa.get("span_analysis") or []):
            ori = sp.get("original_span") or {}
            st_abs = ori.get("start_char")
            en_abs = ori.get("end_char")
            if isinstance(st_abs, int) and isinstance(en_abs, int):
                s = max(0, min(n, st_abs - base))
                e = max(0, min(n, en_abs - base))
                inserts[s] += "<span class='span-box'>"
                inserts[e] += "</span>"

    # 2) token wrappers (use offsets; preserve gaps by assembling char-by-char)
    for i, t in enumerate(toks):
        s_abs = t.get("start_char"); e_abs = t.get("end_char")
        if not isinstance(s_abs, int) or not isinstance(e_abs, int):
            continue
        s_rel = max(0, min(n, s_abs - base))
        e_rel = max(0, min(n, e_abs - base))
        if s_rel >= e_rel:
            continue

        # alpha from your normalizer if available; fallback to |importance|
        imp = float(t.get("importance", 0) or 0.0)
        alpha = (norm[i] if (norm and i < len(norm)) else min(0.85, max(0.0, min(1.0, abs(imp)))))
        color = f"rgba(255,0,0,{alpha:.2f})"  # simple red alpha like your original

        # open token span at start, close at end
        inserts[s_rel] += f"<span class='tk' style='background:{color};'>"
        inserts[e_rel] += "</span>"

    # 3) assemble: for each char, emit any inserts, then the char; finish with tail inserts
    parts: List[str] = []
    parts.append(CSS_BASE)
    header = (
        f"<div class='sbar'><div class='stitle'>Sentence {sa.get('sentence_id', 0)}</div>"
        f"<div class='smeta'>Consensus: <span class='badge'>{cons}</span> | "
        f"Label: {lab} | BERT conf: {conf_txt}</div></div>"
    )
    parts.append(header)
    parts.append("<div class='sentbox overlay-scroll'>")
    for idx, ch in enumerate(sent):
        if inserts[idx]:
            parts.append(inserts[idx])
        parts.append(html.escape(ch))
    if inserts[n]:
        parts.append(inserts[n])
    parts.append("</div>")
    return "".join(parts)

# -------- Tiny adapters for Streamlit / ipywidgets --------
def streamlit_select_sentence(st, production_output: Dict[str, Any],
                              source: Literal["span","kp","auto"] = "auto",
                              key: Optional[str] = None) -> int:
    labels, indices = build_sentence_options(production_output, source)
    choice = st.selectbox("Sentence:", labels, index=0, key=key)
    sid = int(choice.split(":", 1)[0]) if ":" in choice else indices[0]
    return sid

def widgets_select_sentence(widgets, production_output: Dict[str, Any],
                            source: Literal["span","kp","auto"] = "auto"):
    labels, indices = build_sentence_options(production_output, source)
    opts = [(l, i) for l, i in zip(labels, indices)]
    dd = widgets.Dropdown(options=[("Select a sentence.", None)] + opts, description="Sentence:")
    out = widgets.Output()
    return dd, out
