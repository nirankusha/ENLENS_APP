# -*- coding: utf-8 -*-
# st_helpers.py
from typing import Dict, Any, List, Optional, Tuple, Iterable
import html
import math
import streamlit as st

# ---------- small utilities ----------
def toast_info(msg: str):
    st.toast(msg, icon="✅")

def make_sentence_selector(production_output: Dict[str, Any] | None, selected_idx: int):
    """Return (idx, sentence_obj) or (0, None) if nothing to pick."""
    if not production_output:
        return 0, None
    sents = production_output.get("sentence_analyses", []) or []
    if not sents:
        return 0, None

    labels = [f"{i:03d}: {(s.get('sentence_text','') or '').strip()[:80]}"
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

# ---------- panels ----------
def render_coref_panel(coref_groups, production_output, mode="document"):
    """coref_groups: dict(chain_id -> list[{sid, text, score?}])"""
    if not coref_groups:
        st.info("No coreference chains for this sentence.")
        return
    for cid, items in coref_groups.items():
        with st.expander(f"Chain {cid} (members: {len(items)})", expanded=False):
            for it in items:
                sid = it.get("sid")
                txt = (it.get("text", "") or "").replace("\n", " ")[:160]
                score = it.get("score", None)
                meta = None
                try:
                    meta = production_output.get("sentence_analyses", [])[sid] if sid is not None else None
                except Exception:
                    meta = None
                label = meta.get("classification", {}).get("label") if meta else None
                cons = meta.get("classification", {}).get("consensus") if meta else None
                st.markdown(
                    f"- **{sid}** — {html.escape(txt)}"
                    f"{' | ' + str(label) if label else ''}"
                    f"{' | ' + str(cons) if cons else ''}"
                    f"{f' | score={score:.2f}' if isinstance(score, (int, float)) else ''}"
                )

def render_concordance_panel(conc_results):
    if not conc_results:
        st.info("No concordance results.")
        return
    st.caption(f"Results: {len(conc_results)}")
    for r in conc_results[:200]:
        st.markdown(
            f"- `{r.get('path','')}` [{r.get('start','?')}:{r.get('end','?')}] — "
            f"{(r.get('text','') or '')[:160]}"
        )

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