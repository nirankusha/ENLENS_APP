# -*- coding: utf-8 -*-
# st_helpers.py

from typing import Dict, List, Any, Optional, Iterable, Tuple
import math
import html as _html
import streamlit as st


# --------------------------- small utils ---------------------------

def toast_info(msg: str):
    st.toast(msg, icon="✅")


def make_sentence_selector(production_output: Dict[str, Any] | None, selected_idx: int):
    if not production_output:
        return 0, None
    sents = production_output.get("sentence_analyses", []) or []
    if not sents:
        return 0, None

    labels = [f"{i:03d}: {(s.get('sentence_text','') or '').strip()[:80]}"
              for i, s in enumerate(sents)]
    idx = st.selectbox("Pick sentence",
                       range(len(sents)),
                       index=min(selected_idx, len(sents)-1),
                       format_func=lambda i: labels[i])
    return idx, sents[idx]


# --- sentence-change hook: reset chips/terms unless persistence is enabled ---
def reset_terms_on_sentence_change(
    new_sentence_idx: int,
    *,
    key_selected_idx: str = "selected_sentence_idx",
    key_terms: str = "query_terms",
    persist_flag_key: str = "persist_terms_across_sentences"
) -> None:
    """
    Clear selected terms when the sentence changes, unless a boolean
    session-state flag `persist_flag_key` is True.

    Usage in app:
        reset_terms_on_sentence_change(selected_idx)
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


# -------------------- chip sources (span / kp / tokens) --------------------

def _iter_spans(sent: Dict[str, Any]) -> Iterable[Tuple[str, Optional[float]]]:
    """
    Yield (text, score) from multiple possible shapes:
      - sp['original_span']['text'], sp['original_span']['importance']
      - sp['text'] or sp['span'], with 'importance' or 'score'
      - KP runners sometimes stuff phrases here too
    """
    for sp in (sent.get("span_analysis") or []):
        base = sp.get("original_span") or sp
        text = base.get("text") or base.get("span") or sp.get("text") or sp.get("span")
        if not text:
            continue
        score = base.get("importance")
        if score is None:
            score = base.get("score")
        if score is None:
            score = sp.get("importance", sp.get("score"))
        yield str(text).strip(), (None if score is None else float(score))


def _iter_kps(sent: Dict[str, Any]) -> Iterable[Tuple[str, Optional[float]]]:
    # Accept both {'text','score'} and {'kp','score'}
    for kp in (sent.get("keyphrases") or []):
        text = kp.get("text") or kp.get("kp")
        if not text:
            continue
        score = kp.get("score")
        yield str(text).strip(), (None if score is None else float(score))


def _iter_tokens(sent: Dict[str, Any]) -> Iterable[Tuple[str, Optional[float]]]:
    toks = (sent.get("token_analysis") or {}).get("tokens") or []
    for t in toks:
        tok = t.get("token")
        if not tok:
            continue
        score = t.get("importance")
        # IG can return arrays; be defensive
        try:
            score = float(score) if score is not None else None
        except Exception:
            score = None
        yield str(tok).strip(), score


def _uniq_keep_order(items: Iterable[Tuple[str, Optional[float]]]) -> List[Tuple[str, Optional[float]]]:
    seen = set()
    out: List[Tuple[str, Optional[float]]] = []
    for term, score in items:
        key = term.lower()
        if key in seen or not term:
            continue
        seen.add(key)
        out.append((term, score))
    return out


# --------------------------- color mapping ---------------------------

def _score_to_rgba(score: Optional[float]) -> str:
    """
    Map importance to a colored swatch:
      positive -> blue; negative -> red; magnitude controls alpha (opacity).
    If score is None, return a neutral gray.
    """
    if score is None or math.isnan(score):
        return "rgba(150,150,150,0.25)"
    # cap magnitude into [0,1] for opacity
    alpha = max(0.08, min(0.85, abs(score)))
    if score >= 0:
        # blue
        return f"rgba(33,150,243,{alpha:.3f})"   # material blue 500
    # red
    return f"rgba(244,67,54,{alpha:.3f})"       # material red 500


# ----------------------- main renderer (with colors) -----------------------

def render_sentence_text_with_chips(
    sent_obj: Dict[str, Any],
    *,
    candidate_source: str = "span",
    clickable_tokens: bool = False,
) -> Optional[str]:
    """
    Render the sentence text and clickable chips (spans / keyphrases / tokens),
    each accompanied by a small colored swatch encoding contribution score.

    Returns the clicked term (str) or None.
    """
    text = (sent_obj or {}).get("sentence_text", "") or ""
    st.write(text.strip())

    # sentence+mode scoped key prevents collisions when switching rows/modes
    sent_key = f"s{sent_obj.get('sentence_id', 'x')}"
    mode_key = f"m_{candidate_source}_{'tok' if clickable_tokens else 'notok'}"

    # Collect (term, score)
    items: List[Tuple[str, Optional[float]]] = []
    if candidate_source == "span":
        items.extend(_iter_spans(sent_obj))
    elif candidate_source == "kp":
        items.extend(_iter_kps(sent_obj))
    if clickable_tokens:
        items.extend(_iter_tokens(sent_obj))

    chips = _uniq_keep_order(items)

    clicked: Optional[str] = None
    if chips:
        n = len(chips)
        # target ~8 items per column, 2–6 columns
        n_cols = min(6, max(2, (n + 7) // 8))
        cols = st.columns(n_cols)

        for i, (term, score) in enumerate(chips):
            col = cols[i % n_cols]
            with col:
                # colored swatch above the button
                color = _score_to_rgba(score)
                tooltip = "" if score is None else f" (score={score:.4f})"
                st.markdown(
                    f"""
                    <div title="importance{_html.escape(tooltip)}"
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


# --------------------- simple panels (unchanged) ---------------------

def render_coref_panel(coref_groups, production_output, mode="document"):
    """coref_groups: dict(chain_id -> list[ {sid, text, score?} ])"""
    if not coref_groups:
        st.info("No coreference chains for this sentence.")
        return
    for cid, items in coref_groups.items():
        with st.expander(f"Chain {cid} (members: {len(items)})", expanded=False):
            for it in items:
                sid = it.get("sid")
                txt = (it.get("text","") or "")[:160].replace("\n"," ")
                score = it.get("score", None)
                meta = None
                try:
                    if sid is not None:
                        meta = production_output.get("sentence_analyses", [])[int(sid)]
                except Exception:
                    meta = None
                label = (meta or {}).get("classification",{}).get("label")
                cons  = (meta or {}).get("classification",{}).get("consensus")
                st.markdown(
                    f"- **{sid}** — {_html.escape(txt)}"
                    f"{' | ' + str(label) if label else ''}"
                    f"{' | ' + str(cons) if cons else ''}"
                    f"{f' | score={score:.3f}' if isinstance(score,(int,float)) else ''}"
                )


def render_concordance_panel(conc_results):
    if not conc_results:
        st.info("No concordance results.")
        return
    st.caption(f"Results: {len(conc_results)}")
    for r in conc_results[:200]:
        st.markdown(
            f"- `{r.get('path','')}` "
            f"[{r.get('start','?')}:{r.get('end','?')}] — {r.get('text','')[:160]}"
        )


def render_clusters_panel(G, meta, sentence_idx: int, summarize_opts: Dict[str, bool]):
    """List-only summary of communities/clusters for a sentence."""
    summaries = meta.get("summaries") or {}
    comm = meta.get("communities") or {}
    kmeans = meta.get("kmeans")
    torque = meta.get("torque")

    comm_ids = []
    if isinstance(comm, dict):
        for node, cid in comm.items():
            try:
                if int(node) == int(sentence_idx):
                    comm_ids.append(cid)
            except Exception:
                pass
    comm_ids = list(dict.fromkeys(comm_ids))

    if comm_ids:
        st.markdown("**Communities (SciCo):**")
        for cid in comm_ids:
            sm = summaries.get(cid, {})
            parts = []
            if summarize_opts.get("show_representative") and sm.get("representative"):
                parts.append(f"rep: {sm['representative']}")
            if summarize_opts.get("show_xsum") and sm.get("xsum_summary"):
                parts.append(f"xsum: {sm['xsum_summary']}")
            if summarize_opts.get("show_presumm") and sm.get("presumm_top_sent"):
                parts.append(f"presumm: {sm['presumm_top_sent']}")
            st.write(f"- Community {cid}" + (": " + " | ".join(parts) if parts else ""))

    # clusters
    try:
        if isinstance(kmeans, (list, tuple)) and 0 <= sentence_idx < len(kmeans):
            st.markdown(f"**KMeans cluster:** {int(kmeans[sentence_idx])}")
    except Exception:
        pass
    try:
        if isinstance(torque, (list, tuple)) and 0 <= sentence_idx < len(torque):
            st.markdown(f"**Torque cluster:** {int(torque[sentence_idx])}")
    except Exception:
        pass


"""
Created on Mon Aug 25 15:03:05 2025
@author: niran
"""
