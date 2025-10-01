# -*- coding: utf-8 -*-
# st_helpers.py
from typing import Dict, Any, List, Optional, Tuple, Iterable
import html
import math
import streamlit as st

_LINGMESS_REL_COLORS = {
    "MATCH": "🟦",
    "CONTAINS": "🟩",
    "ENT_PRON": "🟨",
    "PRON_PRON_C": "🟪",
    "PRON_PRON_NC": "🟧",
    "OTHER": "⬜",
}

# ---------- small utilities ----------
def toast_info(msg: str):
    st.toast(msg, icon="✅")

def _format_sentence_selector_label(idx: int, sent_obj: Dict[str, Any]) -> str:
    """Compose a compact label for the sentence selector dropdown."""

    def _dedupe(seq: Iterable[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for item in seq:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    sent_obj = sent_obj or {}
    raw_text = str(sent_obj.get("sentence_text") or "")
    snippet_src = " ".join(raw_text.replace("\n", " ").split())
    if len(snippet_src) > 80:
        snippet = snippet_src[:77].rstrip() + "…"
    else:
        snippet = snippet_src

    cls = sent_obj.get("classification") or {}

    def _fmt_field(value: Any) -> str:
        if value is None:
            return "?"
        text = str(value).strip()
        return text or "?"

    label_txt = _fmt_field(cls.get("label"))
    consensus_txt = _fmt_field(cls.get("consensus"))

    conf_val = cls.get("confidence")
    if conf_val is None:
        conf_val = cls.get("score")
    conf_txt: Optional[str] = None
    if isinstance(conf_val, (int, float)):
        try:
            if math.isnan(float(conf_val)):
                conf_txt = None
            else:
                conf_txt = f"{float(conf_val):.2f}"
        except Exception:
            conf_txt = None
    elif isinstance(conf_val, str):
        conf_txt = conf_val.strip() or None

    spans = sent_obj.get("span_analysis") or []
    chain_hits = 0
    chain_ids: List[str] = []
    representatives: List[str] = []
    chain_found = 0
    for sp in spans:
        if not isinstance(sp, dict):
            continue
        ca = sp.get("coreference_analysis") or {}
        if not isinstance(ca, dict):
            continue
        if ca.get("chain_found") or ca.get("chain_id") is not None:
            chain_found += 1

        hits = ca.get("chain_hits")
        if hits is not None:
            try:
                chain_hits += int(float(hits))
            except Exception:
                pass

        chain_id = ca.get("chain_id")
        if chain_id is not None:
            chain_ids.append(str(chain_id))

        rep = ca.get("representative")
        if rep:
            rep_txt = " ".join(str(rep).split())
            if rep_txt:
                representatives.append(rep_txt)

    unique_ids = _dedupe(chain_ids)
    unique_reps = _dedupe(representatives)

    base_count = max(len(unique_ids), chain_found)
    chain_summary = str(base_count)
    if chain_hits:
        chain_summary = f"{chain_summary}({chain_hits})"
    elif unique_reps:
        trimmed = []
        for rep in unique_reps[:2]:
            if len(rep) > 12:
                trimmed.append(rep[:12].rstrip() + "…")
            else:
                trimmed.append(rep)
        if trimmed:
            chain_summary = f"{chain_summary}:{'|'.join(trimmed)}"

    sdg_summary = f"{label_txt}/{consensus_txt}"
    if conf_txt:
        sdg_summary = f"{sdg_summary}@{conf_txt}"

    meta = f"{idx:03d} · chains={chain_summary} · SDG={sdg_summary}"
    return f"{meta}: {snippet}" if snippet else meta



def make_sentence_selector(production_output: Dict[str, Any] | None, selected_idx: int):
    """Return (idx, sentence_obj) or (0, None) if nothing to pick."""
    if not production_output:
        return 0, None
    sents = production_output.get("sentence_analyses", []) or []
    if not sents:
        return 0, None

    labels = [_format_sentence_selector_label(i, s)
              for i, s in enumerate(sents)]
    idx = st.selectbox(
        "Pick sentence",
        range(len(sents)),
        index=min(selected_idx, len(sents)-1),
        format_func=lambda i: labels[i],
        key="__sent_selector"
    )
    return idx, sents[idx]

# --- sentence-change hook: reset chips/terms unless persistence is enabled ---
def reset_terms_on_sentence_change(
    new_sentence_idx: int,
    *,
    key_selected_idx: str = "selected_sentence_idx",
    key_terms: str = "query_terms",
    persist_flag_key: str = "persist_terms_across_sentences",
) -> None:
    """
    Clear selected terms when the sentence changes, unless a boolean
    session-state flag `persist_flag_key` is True.
    """
    ss = st.session_state
    prev_idx_key = f"__prev_{key_selected_idx}"
    prev_idx = ss.get(prev_idx_key, None)
    ss.setdefault(key_terms, [])
    persist = bool(ss.get(persist_flag_key, False))

    if prev_idx is None:
        ss[prev_idx_key] = new_sentence_idx
        return

    if new_sentence_idx != prev_idx:
        if not persist:
            ss[key_terms] = []
        ss[prev_idx_key] = new_sentence_idx

# ---------- robust chip extraction with multiple fallbacks ----------
def _extract_spans(sent_obj: Dict[str, Any]) -> List[Tuple[str, Optional[float]]]:
    """Extract spans from span_analysis with multiple fallback patterns."""
    span_analysis = sent_obj.get("span_analysis", []) or []
    results = []
    
    for sp in span_analysis:
        if not isinstance(sp, dict):
            continue
            
        # Pattern 1: original_span.text (your main data structure)
        original_span = sp.get("original_span")
        if isinstance(original_span, dict):
            text = original_span.get("text", "").strip()
            importance = original_span.get("importance")
            if text:
                score = float(importance) if importance is not None else None
                results.append((text, score))
                continue
        
        # Pattern 2: direct text/span in the span object
        text = sp.get("text") or sp.get("span", "")
        if text:
            text = str(text).strip()
            importance = sp.get("importance") or sp.get("score")
            score = float(importance) if importance is not None else None
            results.append((text, score))
            continue
            
        # Pattern 3: expanded_phrase fallback
        text = sp.get("expanded_phrase", "").strip()
        if text:
            importance = sp.get("importance") or sp.get("score")
            score = float(importance) if importance is not None else None
            results.append((text, score))
    
    return results

def _extract_keyphrases(sent_obj: Dict[str, Any], threshold: float = 0.1) -> List[Tuple[str, Optional[float]]]:
    """Extract keyphrases - could be in span_analysis or separate keyphrases field."""
    results = []
    
    # Check span_analysis first (KPE pipeline puts results there)
    span_results = _extract_spans(sent_obj)
    for text, score in span_results:
        if score is None or score >= threshold:
            results.append((text, score))
    
    # Also check traditional keyphrases field
    keyphrases = sent_obj.get("keyphrases", []) or []
    for kp in keyphrases:
        if isinstance(kp, dict):
            text = kp.get("text") or kp.get("phrase", "")
            if text:
                text = str(text).strip()
                score = kp.get("score")
                if score is None or float(score) >= threshold:
                    results.append((text, float(score) if score is not None else None))
        elif isinstance(kp, (list, tuple)) and len(kp) >= 2:
            text, score = str(kp[0]).strip(), kp[1]
            if score is None or float(score) >= threshold:
                results.append((text, float(score) if score is not None else None))
    
    return results

def _extract_tokens(sent_obj: Dict[str, Any]) -> List[Tuple[str, Optional[float]]]:
    """Extract tokens, filtering out BERT special tokens."""
    token_analysis = sent_obj.get("token_analysis", {}) or {}
    tokens = token_analysis.get("tokens", []) or []
    
    results = []
    for tok in tokens:
        if not isinstance(tok, dict):
            continue
            
        token_text = tok.get("token", "").strip()
        
        # Filter out BERT special tokens and subword pieces
        if (token_text and 
            not token_text.startswith("[") and  # [CLS], [SEP], etc.
            not token_text.startswith("##") and  # subword pieces
            len(token_text) > 1 and  # single chars
            token_text not in {",", ".", "(", ")", ":", ";", "-"}):  # punctuation
            
            importance = tok.get("importance")
            score = float(importance) if importance is not None else None
            results.append((token_text, score))
    
    return results

def _remove_duplicates(items: List[Tuple[str, Optional[float]]]) -> List[Tuple[str, Optional[float]]]:
    """Remove duplicates while preserving order and keeping highest score."""
    seen = {}
    for text, score in items:
        key = text.lower()
        if key not in seen or (score is not None and (seen[key][1] is None or score > seen[key][1])):
            seen[key] = (text, score)
    
    # Preserve original order
    result = []
    seen_keys = set()
    for text, score in items:
        key = text.lower()
        if key not in seen_keys:
            seen_keys.add(key)
            result.append(seen[key])
    
    return result

# ---------- color mapping for importance scores ----------
def _score_to_rgba(score: Optional[float], pos_threshold: float = 0.15, neg_threshold: float = 0.20) -> str:
    """Map importance score to color with thresholds for blue (positive) and red (negative)."""
    if score is None or math.isnan(score):
        return "rgba(150,150,150,0.25)"  # neutral gray
    
    # Determine color based on thresholds
    if score >= pos_threshold:
        # Blue for positive importance above threshold
        alpha = min(0.8, max(0.3, abs(score)))
        return f"rgba(33,150,243,{alpha:.3f})"
    elif score <= -neg_threshold:
        # Red for negative importance below threshold
        alpha = min(0.8, max(0.3, abs(score))) 
        return f"rgba(244,67,54,{alpha:.3f})"
    else:
        # Light gray for scores within threshold range
        return "rgba(150,150,150,0.15)"

# ---------- main renderer ----------
def render_sentence_text_with_chips(
    sent_obj: Dict[str, Any],
    *,
    candidate_source: str = "span",
    clickable_tokens: bool = False,
    pos_threshold: float = 0.15,
    neg_threshold: float = 0.20,
    kpe_top_k: int = 10,
    kpe_threshold: float = 0.10,
    debug: bool = False,
    **unused_kwargs,
) -> Optional[str]:
    """
    Render the sentence text and clickable chips (spans / keyphrases / tokens).
    Returns the clicked term (str) or None.
    """
    text = (sent_obj or {}).get("sentence_text", "") or ""
    st.write(text.strip())

    # sentence-scoped key prefix to prevent cross-sentence widget reuse
    sent_key = f"s{sent_obj.get('sentence_id', 'x')}"
    mode_key = f"m_{candidate_source}_{'tok' if clickable_tokens else 'notok'}"

    # Extract chips based on source
    chips: List[Tuple[str, Optional[float]]] = []
    
    if debug:
        st.write(f"**DEBUG:** Extracting chips for candidate_source='{candidate_source}', clickable_tokens={clickable_tokens}")
    
    if candidate_source == "span":
        span_chips = _extract_spans(sent_obj)
        chips.extend(span_chips)
        if debug:
            st.write(f"**DEBUG:** Extracted {len(span_chips)} spans: {[text for text, _ in span_chips[:3]]}")
    
    elif candidate_source == "kp":
        kp_chips = _extract_keyphrases(sent_obj, threshold=kpe_threshold)
        # Sort by score and take top-k
        kp_chips_sorted = sorted(kp_chips, key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
        chips.extend(kp_chips_sorted[:kpe_top_k])
        if debug:
            st.write(f"**DEBUG:** Extracted {len(kp_chips)} keyphrases, kept top {len(chips)}: {[text for text, _ in chips[:3]]}")
    
    if clickable_tokens:
        token_chips = _extract_tokens(sent_obj)
        chips.extend(token_chips)
        if debug:
            st.write(f"**DEBUG:** Added {len(token_chips)} tokens")
    
    # Remove duplicates while preserving order
    chips = _remove_duplicates(chips)
    
    if debug:
        st.write(f"**DEBUG:** Final chip count after deduplication: {len(chips)}")
        if chips:
            st.write(f"**DEBUG:** Final chips: {[f'{text} ({score})' for text, score in chips[:5]]}")

    clicked = None
    if chips:
        # Create clickable buttons with colored swatches
        n = len(chips)
        n_cols = min(6, max(2, (n + 7) // 8))
        cols = st.columns(n_cols)

        for i, (term, score) in enumerate(chips):
            col = cols[i % n_cols]
            with col:
                # Colored swatch above the button
                color = _score_to_rgba(score, pos_threshold, neg_threshold)
                tooltip = f"score={score:.4f}" if score is not None else "no score"
                
                st.markdown(
                    f"""
                    <div title="importance: {html.escape(tooltip)}"
                         style="width:100%;height:6px;border-radius:6px;margin:4px 0 2px 0;background:{color};"></div>
                    """,
                    unsafe_allow_html=True,
                )
                
                key = f"chip_{sent_key}_{mode_key}_{i}"
                if st.button(term, key=key):
                    clicked = term
    else:
        st.caption("No spans / keyphrases / tokens found for this sentence.")

    return clicked

import html
from collections import Counter
import streamlit as st

def render_coref_panel(coref_groups, production_output, mode: str = "document"):
    """
    coref_groups: {chain_id -> [ {sid, text, mentions:[{text,start_char,end_char,tag?}, ...]}, ... ]}
    production_output: full production dict (used to look up chain 'edges' and global 'mentions')
    """
    if not coref_groups:
        st.info("No coreference chains for this sentence.")
        return

    TAG_COLORS = {
        "PRON-PRON-C": "#6e8efb",
        "PRON_PRON_C": "#6e8efb",
        "PRON-PRON-NC": "#a1a1a1",
        "PRON_PRON_NC": "#a1a1a1",
        "ENT-PRON":     "#56b881",
        "ENT_PRON":     "#56b881",
        "MATCH":        "#e6b422",
        "CONTAINS":     "#e86a5f",
        "OTHER":        "#999999",
        "NO_RELATION":  "#999999",
        "NOT_COREF":    "#999999",
        None:           "#cccccc",
    }

    # Lookup for full chains (to access 'edges' and chain-level mentions)
    chains = (production_output.get("coreference_analysis") or {}).get("chains") or []
    chain_by_id = {int(c.get("chain_id", -1)): c for c in chains if isinstance(c, dict)}

    for cid, items in coref_groups.items():
        chain = chain_by_id.get(int(cid))
        # Summarize relationships if present
        rel_chips_html = ""
        edge_rows = []
        
        if chain:
            edges = chain.get("edges") or []
            if edges:
                
                def _canon_tag(tag_val: Any) -> str:
                    tag_txt = str(tag_val).strip().upper() if tag_val is not None else ""
                    tag_txt = tag_txt.replace("-", "_")
                    return tag_txt or "OTHER"

                def _edge_tag(edge: Dict[str, Any]) -> str:
                    if isinstance(edge, dict):
                        for key in ("tag", "relation", "lingmess_relation", "heuristic"):
                            if edge.get(key):
                                return _canon_tag(edge.get(key))
                    return "OTHER"

                counts = Counter(_edge_tag(e) for e in edges)
                chips = []
                for tag, cnt in counts.items():
                    color = TAG_COLORS.get(tag, TAG_COLORS.get(tag.replace("_", "-"), "#999999"))
                    chips.append(
                        f"<span style='display:inline-block;padding:2px 6px;border-radius:6px;"
                        f"background:{color};color:white;font-size:11px;margin-right:6px;'>{html.escape(tag)}: {cnt}</span>"
                    )
                rel_chips_html = "".join(chips)

                # Prepare a short list of edge lines using chain-level mentions by index
                mlist = chain.get("mentions") or []
                max_show = min(40, len(edges))
                for e in edges[:max_show]:
                    i = e.get("antecedent", e.get("i"))
                    j = e.get("anaphor", e.get("j"))
                    tag = _edge_tag(e)
                    try:
                        i = int(i); j = int(j)
                    except Exception:
                        continue
                    m1 = mlist[i]["text"] if isinstance(i, int) and 0 <= i < len(mlist) else "?"
                    m2 = mlist[j]["text"] if isinstance(j, int) and 0 <= j < len(mlist) else "?"
                    edge_rows.append((tag, m1, m2, e.get("heuristic"), e.get("relation_source"), e.get("lingmess_relation"), e.get("lingmess_score")))

        with st.expander(f"Chain {cid} • sentences: {len(items)}", expanded=False):
            # Relationships summary chips (if any)
            if rel_chips_html:
                st.markdown(rel_chips_html, unsafe_allow_html=True)

            # Per-sentence mentions
            for row in sorted(items, key=lambda r: r.get("sid", 10**9)):
                # Resolve sentence text safely
                if "text" in row and isinstance(row["text"], str):
                    sent_text = row["text"]
                else:
                    sents = production_output.get("sentence_analyses") or []
                    sid = row.get("sid")
                    sent_text = (sents[sid].get("sentence_text", "")
                                 if isinstance(sid, int) and 0 <= sid < len(sents) else "")
                st.markdown(
                    f"**Sentence {row.get('sid', '?')}** — {html.escape(sent_text.replace(chr(10),' ')[:200])}"
                )

                # Mentions with tag pill
                for m in row.get("mentions", []):
                    tag = m.get("tag")
                    color = TAG_COLORS.get(tag, "#999999")
                    mention_text = html.escape(m.get("text", "") or "")
                    start = m.get("start_char", m.get("start", "?"))
                    end   = m.get("end_char",   m.get("end",   "?"))
                    st.markdown(
                        f"<span style='display:inline-block;padding:2px 6px;border-radius:6px;"
                        f"background:{color};color:white;font-size:11px;margin-right:6px;'>{html.escape(tag or '--')}</span>"
                        f"<code>{mention_text}</code> [{start}:{end}]",
                        unsafe_allow_html=True,
                    )

            # Optional: show a compact edge list
            if edge_rows:
                with st.expander("Relationships (pairs)", expanded=False):
                    for tag, m1, m2, heuristic_rel, relation_source, lingmess_rel, lingmess_score in edge_rows:
                        color = TAG_COLORS.get(tag, TAG_COLORS.get(tag.replace("_", "-"), "#999999"))
                        extras = []
                        if lingmess_rel and lingmess_rel != tag:
                            extras.append(f"LingMess={html.escape(str(lingmess_rel))}")
                        if heuristic_rel and heuristic_rel != tag:
                            extras.append(f"heuristic={html.escape(str(heuristic_rel))}")
                        if relation_source and relation_source != "lingmess":
                            extras.append(f"source={html.escape(str(relation_source))}")
                        if isinstance(lingmess_score, (int, float)):
                            extras.append(f"score={lingmess_score:.3f}")
                        extra_txt = f" ({', '.join(extras)})" if extras else ""
                        st.markdown(
                            f"<span style='display:inline-block;padding:2px 6px;border-radius:6px;"
                            f"background:{color};color:white;font-size:11px;margin-right:6px;'>{html.escape(tag)}</span>"
                            f"“{html.escape(m1)}” ↔ “{html.escape(m2)}”{extra_txt}",
                            unsafe_allow_html=True,
                        )

def render_concordance_panel(conc_results):
    """Render concordance rows along with backend metadata."""
    if not conc_results:
        st.info("No concordance results.")
        return
    
    rows: List[Dict[str, Any]]
    meta: Dict[str, Any] = {}
    terms: List[str] = []

    if isinstance(conc_results, dict):
        rows = list(conc_results.get("rows", []) or [])
        meta = conc_results.get("meta") or {}
        terms = list(conc_results.get("terms", []) or [])
    else:
        # Legacy adapters may still return a list – keep rendering them.
        rows = list(conc_results)

    if not rows:
        st.info("No concordance matches found.")
    summary_bits: List[str] = [f"Results: {len(rows)}"]
    if terms:
        summary_bits.append("Terms: " + ", ".join(map(str, terms)))

    mode = meta.get("mode") if isinstance(meta, dict) else None
    if mode:
        summary_bits.append(f"Mode: {mode}")

    vector_backend = meta.get("vector_backend") if isinstance(meta, dict) else None
    if vector_backend:
        summary_bits.append(f"Vector backend: {vector_backend}")

    faiss_used = meta.get("faiss_used") if isinstance(meta, dict) else None
    if faiss_used is not None:
        summary_bits.append(f"FAISS: {'yes' if faiss_used else 'no'}")

    total_candidates = meta.get("total_candidates") if isinstance(meta, dict) else None
    if isinstance(total_candidates, int):
        summary_bits.append(f"Candidates scanned: {total_candidates}")

    if summary_bits:
        st.caption(" • ".join(summary_bits))

    query_text = meta.get("query_text") if isinstance(meta, dict) else None
    if query_text:
        st.caption(f"Query: `{query_text}`")

    for r in rows[:200]:
        st.markdown(
            f"- `{r.get('path','')}` [{r.get('start','?')}:{r.get('end','?')}] — "
            f"{(r.get('text','') or '')[:160]}"
        )


def _normalize_spacing(tokens: list[str]) -> str:
    """
    Join tokens into a readable phrase:
   - no space before , . ; : ? ! )
    - no space after ( or opening quotes
    """
    no_space_before = {",", ".", ";", ":", "?", "!", ")", "’", '"', "”"}
    no_space_after  = {"(", "‘", '"', "“"}
    out: list[str] = []
    for i, t in enumerate(tokens):
        if i == 0:
            out.append(t)
            continue
        prev = out[-1]
        if t in no_space_before:
            out[-1] = prev + t
        elif prev in no_space_after:
            out[-1] = prev + t
        else:
            out.append(" " + t)
    return "".join(out)

def _ensure_query_builder():
    ss = st.session_state
    ss.setdefault("query_builder", {
        "active_tokens": [],     # tokens clicked but not committed
        "current_phrase": "",    # live preview of the phrase under construction
        "phrases": [],           # committed phrases (list[str])
        "ops": [],               # future: operators between phrases (AND/OR/NEAR/…)
        "constraints": {},       # future: flexiconc constraints
    })
    # Back-compat shim: keep query_terms in sync with committed phrases
    ss.setdefault("query_terms", ss["query_builder"]["phrases"])
    return ss["query_builder"]

def handle_clicked_term(term: str):
    qb = _ensure_query_builder()
    # Treat multi-word chunks as single units for now
    qb["active_tokens"].append(term)
    qb["current_phrase"] = _normalize_spacing(qb["active_tokens"])

def commit_current_phrase():
    qb = _ensure_query_builder()
    phrase = qb["current_phrase"].strip()
    if phrase and phrase not in qb["phrases"]:
        qb["phrases"].append(phrase)
        st.session_state["query_terms"] = qb["phrases"]  # sync alias
    # FIX: this line was truncated in your file
    qb["active_tokens"].clear()
    qb["current_phrase"] = ""

def undo_last_token():
    qb = _ensure_query_builder()
    if qb["active_tokens"]:
        qb["active_tokens"].pop()
        qb["current_phrase"] = _normalize_spacing(qb["active_tokens"])
        
def clear_phrase_builder():
    qb = _ensure_query_builder()
    qb["active_tokens"].clear()
    qb["current_phrase"] = ""

def remove_phrase(i: int):
    qb = _ensure_query_builder()
    if 0 <= i < len(qb["phrases"]):
        qb["phrases"].pop(i)
        st.session_state["query_terms"] = qb["phrases"]

def compile_flexiconc_query(
    phrases: list[str],
    ops: list[str] | None = None,
    constraints: dict | None = None,
) -> str:
    """
    Hook for future compound FlexiConc queries.
    For now: produce a simple quoted-conjunction string: "ph1" "ph2" ...
    """
    if not phrases:
        return ""
    return " ".join(f"\"{p}\"" for p in phrases)

# ==================== Top-K chip bar ====================

def render_topk_chip_bar(
    sent_obj: Dict[str, Any],
    *,
    topk_tokens: int = 8,
    topk_spans: int = 6,
    min_abs_importance: float = 0.10,
    pos_threshold: float = 0.15,
    neg_threshold: float = 0.20,
    candidate_source: str = "span",  # "span" | "kp" | "auto"
) -> Optional[str]:
    """
    Compact chip bar:
      • top-K tokens by |importance|
      • top-M spans by |importance|
    Returns clicked term (str) or None.
    """
    clicked: Optional[str] = None

    # tokens
    toks = _extract_tokens(sent_obj)  # List[(text, score)]
    toks = [(t, s) for (t, s) in toks if s is None or abs(float(s)) >= float(min_abs_importance)]
    toks_sorted = sorted(toks, key=lambda x: (0.0 if x[1] is None else abs(float(x[1]))), reverse=True)[:topk_tokens]

    # spans (from span_analysis OR KPE depending on candidate_source)
    spans = _extract_spans(sent_obj)  # List[(text, score)]
    spans = [(t, s) for (t, s) in spans if s is None or abs(float(s)) >= float(min_abs_importance)]
    spans_sorted = sorted(spans, key=lambda x: (0.0 if x[1] is None else abs(float(x[1]))), reverse=True)[:topk_spans]
    # Header
    st.caption("🎯 Top-K candidates (click to add to phrase)")

    # Render token chips
    if toks_sorted:
        st.write("**Tokens**")
        cols = st.columns(min(6, max(2, len(toks_sorted))))
        for i, (term, score) in enumerate(toks_sorted):
            with cols[i % len(cols)]:
                color = _score_to_rgba(score, pos_threshold, neg_threshold)
                tooltip = f"score={score:.4f}" if isinstance(score, (int, float)) else "no score"
                st.markdown(
                    f"""
                    <div title="{html.escape(tooltip)}" style="margin-bottom:4px;">
                      <div style="width:100%;height:4px;background:{color};border-radius:4px;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(term, key=f"tok_chip_{i}"):
                    clicked = term

    # Render span chips
    if spans_sorted:
        st.write("**Spans/Keyphrases**")
        cols = st.columns(min(6, max(2, len(spans_sorted))))
        for i, (term, score) in enumerate(spans_sorted):
            with cols[i % len(cols)]:
                color = _score_to_rgba(score, pos_threshold, neg_threshold)
                tooltip = f"score={score:.4f}" if isinstance(score, (int, float)) else "no score"
                st.markdown(
                    f"""
                    <div title="{html.escape(tooltip)}" style="margin-bottom:4px;">
                      <div style="width:100%;height:4px;background:{color};border-radius:4px;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(term, key=f"span_chip_{i}"):
                    clicked = term

    return clicked

def render_clickable_token_strip(
    sent_obj: Dict[str, Any],
    *,
    max_per_row: int = 10,
    pos_threshold: float = 0.15,
    neg_threshold: float = 0.20,
) -> Optional[str]:
    """
    Render *all* tokens as small buttons in wrapped rows (readable, clickable).
    Returns clicked token or None.
    """
    toks = _extract_tokens(sent_obj)  # List[(text, score)]
    if not toks:
        return None
    clicked = None
    st.caption("🖱️ Clickable tokens")
    # chunk into rows
    row = []
    for idx, (term, score) in enumerate(toks):
        row.append((term, score, idx))
        if len(row) == max_per_row or idx == len(toks) - 1:
            cols = st.columns(len(row))
            for i, (term_i, score_i, abs_i) in enumerate(row):
                with cols[i]:
                    color = _score_to_rgba(score_i, pos_threshold, neg_threshold)
                    tooltip = f"score={score_i:.4f}" if isinstance(score_i, (int,float)) else "no score"
                    st.markdown(
                        f'<div title="{html.escape(tooltip)}" '
                        f'style="margin-bottom:4px;"><div '
                        f'style="width:100%;height:3px;background:{color};border-radius:4px;"></div></div>',
                        unsafe_allow_html=True
                    )
                    if st.button(term_i, key=f"tokbtn_{abs_i}"):
                        clicked = term_i
            row = []
    return clicked

def render_clusters_panel(G, meta, sentence_idx: int, summarize_opts: Dict[str, bool]):
    """List-only summary of communities/clusters for a sentence."""
    summaries = meta.get("summaries") or {}
    comm = meta.get("communities") or {}
    kmeans = meta.get("kmeans")
    torque = meta.get("torque")

    # list communities that contain sentence_idx
    comm_ids = []
    if isinstance(comm, dict):
        for node, cid in comm.items():
            try:
                if int(node) == int(sentence_idx):
                    comm_ids.append(cid)
            except Exception:
                continue
    comm_ids = list(dict.fromkeys(comm_ids))

    if comm_ids:
        st.markdown("**Communities (SciCo):**")
        for cid in comm_ids:
            sm = summaries.get(cid, {})
            line = []
            if summarize_opts.get("show_representative") and sm.get("representative"):
                line.append(f"rep: {sm['representative']}")
            if summarize_opts.get("show_xsum") and sm.get("xsum_summary"):
                line.append(f"xsum: {sm['xsum_summary']}")
            if summarize_opts.get("show_presumm") and sm.get("presumm_top_sent"):
                line.append(f"presumm: {sm['presumm_top_sent']}")
            st.write(f"- Community {cid} " + (" | " + " | ".join(line) if line else ""))

    # clusters: show labels for this node if available
    try:
        if kmeans is not None:
            st.markdown(f"**KMeans cluster:** {int(kmeans[sentence_idx])}")
    except Exception:
        pass
    try:
        if torque is not None:
            st.markdown(f"**Torque cluster:** {int(torque[sentence_idx])}")
    except Exception:
        pass