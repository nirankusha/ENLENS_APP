# -*- coding: utf-8 -*-
# st_helpers.py
from typing import Dict, Any, List, Optional
import html
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

# ---------- chip sources ----------
def _chips_from_spans(sent_obj: Dict[str, Any]) -> List[str]:
    """Extract spans from the span_analysis field"""
    out = []
    span_analysis = sent_obj.get("span_analysis", []) or []
    
    for sp in span_analysis:
        if not isinstance(sp, dict):
            continue
            
        # Get text from original_span.text (your data structure)
        original_span = sp.get("original_span", {})
        if isinstance(original_span, dict):
            text = original_span.get("text", "").strip()
            if text:
                out.append(text)
    
    # Remove duplicates while preserving order
    seen = set()
    return [c for c in out if c and not (c in seen or seen.add(c))]

def _chips_from_kps(sent_obj: Dict[str, Any], top_k=10, thr=0.1) -> List[str]:
    """For KP mode, also look in span_analysis since KPE pipeline puts data there"""
    # In your system, keyphrases are also in span_analysis
    span_analysis = sent_obj.get("span_analysis", []) or []
    pairs = []
    
    for sp in span_analysis:
        if not isinstance(sp, dict):
            continue
            
        original_span = sp.get("original_span", {})
        if isinstance(original_span, dict):
            text = original_span.get("text", "").strip()
            importance = float(original_span.get("importance", 0.0))
            
            if text and importance >= thr:
                pairs.append((text, importance))
    
    # Sort by importance/score descending
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in pairs[:top_k]]

def _chips_from_tokens(sent_obj: Dict[str, Any]) -> List[str]:
    """Extract tokens, filtering out BERT special tokens"""
    token_analysis = sent_obj.get("token_analysis", {}) or {}
    tokens = token_analysis.get("tokens", []) or []
    
    out = []
    for tok in tokens:
        if not isinstance(tok, dict):
            continue
            
        token_text = tok.get("token", "").strip()
        
        # Filter out BERT special tokens and subword pieces
        if (token_text and 
            not token_text.startswith("[") and  # [CLS], [SEP], etc.
            not token_text.startswith("##") and  # subword pieces
            len(token_text) > 1 and  # single chars like "-"
            token_text not in {",", ".", "(", ")", ":", ";"}):  # punctuation
            out.append(token_text)
    
    # Remove duplicates while preserving order
    seen = set()
    return [c for c in out if c and not (c in seen or seen.add(c))]

# ---------- main renderer ----------
# Debug function for troubleshooting
def debug_sentence_data(sent_obj: Dict[str, Any], st_module):
    """Debug function to show what's in the sentence object"""
    if st_module:
        with st_module.expander("🔍 DEBUG: Sentence data structure", expanded=False):
            st_module.write("**Available keys:**")
            st_module.code(list(sent_obj.keys()))
            
            st_module.write("**span_analysis content:**")
            span_data = sent_obj.get("span_analysis", [])
            if span_data:
                st_module.write(f"Found {len(span_data)} span items")
                for i, item in enumerate(span_data[:3]):  # Show first 3
                    st_module.write(f"Item {i}: {list(item.keys()) if isinstance(item, dict) else type(item)}")
                    if isinstance(item, dict) and "original_span" in item:
                        st_module.code(f"original_span: {item['original_span']}")
            else:
                st_module.write("No span_analysis data")
            
            st_module.write("**token_analysis content:**")
            token_data = sent_obj.get("token_analysis", {}).get("tokens", [])
            if token_data:
                st_module.write(f"Found {len(token_data)} tokens")
                st_module.code(f"Sample token: {token_data[0] if token_data else 'None'}")
            else:
                st_module.write("No token data")

# Main render function with debug info
def render_sentence_text_with_chips(
    sent_obj: Dict[str, Any],
    *,
    candidate_source: str = "span",
    clickable_tokens: bool = False,
    pos_threshold: float = 0.15,
    neg_threshold: float = 0.20,
    kpe_top_k: int = 10,
    kpe_threshold: float = 0.10,
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

    # DIRECT CHIP EXTRACTION (bypass the separate functions)
    chips: List[str] = []
    
    if candidate_source == "span":
        # Extract from span_analysis directly
        span_analysis = sent_obj.get("span_analysis", []) or []
        st.write(f"**DEBUG:** Found {len(span_analysis)} spans in span_analysis")
        
        for i, sp in enumerate(span_analysis):
            if isinstance(sp, dict):
                original_span = sp.get("original_span", {})
                if isinstance(original_span, dict):
                    text_content = original_span.get("text", "").strip()
                    if text_content:
                        chips.append(text_content)
                        st.write(f"**DEBUG:** Added span {i}: '{text_content}'")
    
    elif candidate_source == "kp":
        # For KP mode, also look in span_analysis (since your system puts KP data there)
        span_analysis = sent_obj.get("span_analysis", []) or []
        st.write(f"**DEBUG:** Found {len(span_analysis)} items for KP extraction")
        
        for i, sp in enumerate(span_analysis):
            if isinstance(sp, dict):
                original_span = sp.get("original_span", {})
                if isinstance(original_span, dict):
                    text_content = original_span.get("text", "").strip()
                    importance = float(original_span.get("importance", 0.0))
                    if text_content and importance >= kpe_threshold:
                        chips.append(text_content)
                        st.write(f"**DEBUG:** Added KP {i}: '{text_content}' (importance: {importance})")
    
    if clickable_tokens:
        # Extract tokens directly
        token_analysis = sent_obj.get("token_analysis", {}) or {}
        tokens = token_analysis.get("tokens", []) or []
        st.write(f"**DEBUG:** Found {len(tokens)} tokens")
        
        for i, tok in enumerate(tokens[:10]):  # Just first 10 for testing
            if isinstance(tok, dict):
                token_text = tok.get("token", "").strip()
                # Filter out special tokens
                if (token_text and 
                    not token_text.startswith("[") and 
                    not token_text.startswith("##") and
                    len(token_text) > 1):
                    chips.append(token_text)
                    if i < 5:  # Show debug for first 5
                        st.write(f"**DEBUG:** Added token {i}: '{token_text}'")

    # Remove duplicates while preserving order
    seen = set()
    clean_chips = []
    for c in chips:
        if c and c not in seen:
            seen.add(c)
            clean_chips.append(c)
    chips = clean_chips

    st.write(f"**DEBUG:** Final chip count: {len(chips)}")
    if chips:
        st.write(f"**DEBUG:** Final chips: {chips[:5]}...")  # Show first 5

    clicked = None
    if chips:
        # Create clickable buttons
        n = len(chips)
        n_cols = min(6, max(2, (n + 7) // 8))
        cols = st.columns(n_cols)

        for i, term in enumerate(chips):
            col = cols[i % n_cols]
            with col:
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
