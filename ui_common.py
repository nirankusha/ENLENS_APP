"""
ui_common.py - shared UI helpers (standalone)

`production_output` schema (normalized):
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
def _count_spans_and_coref(sa_item: Dict[str, Any]) -> Tuple[int, int]:
    spans = sa_item.get("span_analysis", []) or []
    total = len(spans)
    with_coref = sum(1 for s in spans if (s.get("coreference_analysis") or {}).get("chain_found"))
    return total, with_coref

def format_sentence_option(sa_item: Dict[str, Any], source: Literal["span","kp","auto"] = "auto") -> str:
    i = sa_item.get("sentence_id", 0)
    sent = (sa_item.get("sentence_text") or "").replace("\n", " ")
    snippet = (sent[:60] + ("..." if len(sent) > 60 else ""))
    cls = sa_item.get("classification", {})
    consensus = cls.get("consensus", "?")
    b_conf = cls.get("confidence", None)
    b_conf_txt = f"{b_conf:.2f}" if isinstance(b_conf, (int, float)) else "?"
    if source == "auto":
        n = len(sa_item.get("span_analysis") or [])
        source = "span" if n and any("importance" in (s.get("original_span") or {}) for s in sa_item.get("span_analysis") or []) else "kp"
    code = "SP" if source == "span" else "KP"
    total, with_coref = _count_spans_and_coref(sa_item)
    return f"{i}: {consensus} | BERT={b_conf_txt} | {code}={total}({with_coref}) | {snippet}"

def build_sentence_options(production_output: Dict[str, Any],
                           source: Literal["span","kp","auto"] = "auto") -> Tuple[List[str], List[int]]:
    items = production_output.get("sentence_analyses") or []
    labels = [format_sentence_option(sa, source=source) for sa in items]
    indices = [int(sa.get("sentence_id", idx)) for idx, sa in enumerate(items)]
    return labels, indices

# -------- HTML overlay --------
CSS_BASE = """
<style>
.sbar {padding:6px 10px;border:1px solid #e5e7eb;border-radius:10px;background:#f7fafc;margin:6px 0}
.stitle {font-weight:600}
.smeta {color:#555;font-size:12px;margin-top:4px}
.sentbox {padding:10px 12px;border:1px solid #e5e7eb;border-radius:10px;background:#fff}
.tk {padding:1px 2px;border-radius:4px}
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

def render_sentence_overlay(production_output: Dict[str, Any], sentence_id: int,
                            highlight_coref: bool = True,
                            box_spans: bool = True) -> str:
    items = production_output.get("sentence_analyses") or []
    lookup = {int(sa.get("sentence_id", i)): sa for i, sa in enumerate(items)}
    sa = lookup.get(int(sentence_id))
    if not sa:
        return "<div>Sentence not found.</div>"

    sent = sa.get("sentence_text", "")
    cls = sa.get("classification", {})
    cons = html.escape(str(cls.get("consensus", "?")))
    lab = html.escape(str(cls.get("label", "?")))
    conf = cls.get("confidence", None)
    conf_txt = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"

    # token overlay
    toks = (sa.get("token_analysis") or {}).get("tokens") or []
    norm = _normalize_importances(toks)

    # span boxes (absolute -> sentence-local)
    marks = ["" for _ in range(len(sent) + 1)]
    if box_spans:
        for sp in (sa.get("span_analysis") or []):
            st = (sp.get("original_span") or {}).get("start_char")
            en = (sp.get("original_span") or {}).get("end_char")
            if isinstance(st, int) and isinstance(en, int):
                base = int(sa.get("doc_start", 0))
                s0, s1 = st - base, en - base
                if 0 <= s0 < len(marks): marks[s0] += "<span class='span-box'>"
                if 0 <= s1 <= len(marks): marks[s1] += "</span>"

    # render tokens with opacity mapped from importance; preserve spacing using offsets
    html_tokens: List[str] = []
    for i, t in enumerate(toks):
        s = int(t.get("start_char", -1)); e = int(t.get("end_char", -1))
        base = int(sa.get("doc_start", 0))
        s_rel, e_rel = s - base, e - base
        piece = html.escape(sent[s_rel:e_rel]) if 0 <= s_rel < e_rel <= len(sent) else html.escape(str(t.get("token","")))
        alpha = norm[i] if i < len(norm) else 0.0
        style = f"background: rgba(255,0,0,{alpha:.2f});"
        html_tokens.append(f"<span class='tk' style='{style}'>{piece}</span>")

    sent_html = "".join(html_tokens) if html_tokens else html.escape(sent)
    header = (
        f"<div class='sbar'><div class='stitle'>Sentence {sa.get('sentence_id', 0)}</div>"
        f"<div class='smeta'>Consensus: <span class='badge'>{cons}</span> · Label: {lab} · BERT conf: {conf_txt}</div></div>"
    )
    return CSS_BASE + header + f"<div class='sentbox'>{marks[0]}{sent_html}{marks[-1]}</div>"

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
