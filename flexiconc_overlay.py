
"""
flexiconc_overlay.py
--------------------
Interactive inline-overlay concordance selector for FlexiConc-style SQLite exports.

Key features
- Renders sentence text with clickable buttons *only* for words that intersect top spans from
  your classifier (result['production_output']['sentence_analyses'][i]['span_analysis']).
- Provides AND/OR multi-term querying against a precomputed inverted index of the FlexiConc
  sentence store (loaded from the DB paths embedded in the export's "spans_file" table).
- Exposes a callback hook `on_selection(terms, matches)` that fires on each change, so
  downstream modules can trigger clustering/graph building immediately.
- Returns a small "state" object you can also poll programmatically.

Works with the "production_output" produced by your SpanBERT/IG pipelines, i.e. a dict with:
  production_output['sentence_analyses'] = [
     {'sentence_id': int, 'sentence_text': str, 'doc_start': int, 'doc_end': int,
      'span_analysis': [{'text': '...', 'coords': (g0, g1), 'importance': float, ...}, ...]
     },
     ...
  ]
"""

from __future__ import annotations
import sqlite3, re, os
from pathlib import Path
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Optional, Any
import ipywidgets as W
from IPython.display import display, HTML

__all__ = ["launch_inline_overlay_widget", "ConcordanceSelectionState"]

# ---------- schema helpers ----------
def _first_present(d, *cands):
    for c in cands:
        if c in d: return c
    return None

def discover_schema(db_path: str):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    sent_tbl = next((t for t in ("spans_s","spans","segments") if t in tables), None)
    file_tbl = next((t for t in ("spans_file","files","documents","texts") if t in tables), None)
    if not sent_tbl:
        con.close()
        raise SystemExit(f"No sentence table found. Tables: {sorted(tables)}")

    cur.execute(f"PRAGMA table_info({sent_tbl})")
    scols = {r[1] for r in cur.fetchall()}
    s_id   = _first_present(scols, "id", "rowid", "_id")
    s_start= _first_present(scols, "start", "begin", "start_char")
    s_end  = _first_present(scols, "end", "stop", "end_char")
    s_file = _first_present(scols, "file_id", "doc_id", "fid", "file")

    if file_tbl:
        cur.execute(f"PRAGMA table_info({file_tbl})")
        fcols = {r[1] for r in cur.fetchall()}
        f_id   = _first_present(fcols, "id", "rowid", "_id")
        f_path = _first_present(fcols, "path", "filepath", "filename", "name")
    else:
        f_id=f_path=None

    con.close()
    return dict(sent_tbl=sent_tbl, file_tbl=file_tbl,
                s_cols=dict(id=s_id,start=s_start,end=s_end,file_id=s_file),
                f_cols=dict(id=f_id,path=f_path),
                tables=sorted(tables))

# ---------- load sentences once + build inverted index ----------
def load_sentences_and_index(db_path: str, limit=20000):
    sch = discover_schema(db_path)
    sent_tbl = sch["sent_tbl"]
    file_tbl = sch["file_tbl"]
    s_id, s_start, s_end, s_file = (sch["s_cols"][k] for k in ("id","start","end","file_id"))
    f_id, f_path = (sch["f_cols"].get("id"), sch["f_cols"].get("path"))

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    if file_tbl and s_file and f_id and f_path:
        cur.execute(f"""
          SELECT s.{s_id}, s.{s_start}, s.{s_end}, f.{f_path}
          FROM {sent_tbl} s JOIN {file_tbl} f ON s.{s_file} = f.{f_id}
          LIMIT {int(limit)}
        """)
    else:
        # fallback: try to read path from sentence table (rare)
        cur.execute(f"PRAGMA table_info({sent_tbl})")
        scols = {r[1] for r in cur.fetchall()}
        s_path = _first_present(scols, "path","filepath","filename","name")
        if not s_path:
            con.close()
            raise SystemExit("Cannot resolve file path column; ensure spans_file exists with a 'path' column.")
        cur.execute(f"SELECT {s_id}, {s_start}, {s_end}, {s_path} FROM {sent_tbl} LIMIT {int(limit)}")

    rows = cur.fetchall()
    con.close()

    # Build sentence records and inverted index
    sent_records = {}   # sid -> {text, path, start, end}
    inv = defaultdict(set)  # word(lower) -> set(sid)
    word_pat = re.compile(r"\w[\w-]*", flags=re.UNICODE)

    for sid, s0, s1, path in rows:
        p = Path(path)
        if not p.exists():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        sent = txt[s0:s1]
        if not sent:
            continue
        sent_records[sid] = dict(text=sent, path=str(p), start=s0, end=s1)
        for w in word_pat.findall(sent):
            inv[w.lower()].add(sid)

    return sent_records, inv

# ---------- render helpers ----------
def bold_all_terms(s: str, terms):
    if not terms: return s
    # Sort longer terms first to avoid partial masking
    terms_sorted = sorted(set(t.lower() for t in terms), key=len, reverse=True)
    def repl(m):
        return f"<b>{m.group(0)}</b>"
    out = s
    for t in terms_sorted:
        out = re.sub(rf"(?i)\b{re.escape(t)}\b", repl, out)
    return out

def make_inline_sentence_buttons(sentence_text, span_word_positions, on_toggle_factory):
    """
    sentence_text: the full sentence string
    span_word_positions: list of tuples [(start_idx, end_idx, word), ...] for words that belong to spans
    on_toggle_factory: fn(word) -> observer callback
    """
    # Build a char mask: for each char, is it inside a "span-word"?
    slot = [[] for _ in range(len(sentence_text)+1)]  # collect words starting at position
    for s,e,w in span_word_positions:
        s = max(0, min(s, len(sentence_text)))
        e = max(0, min(e, len(sentence_text)))
        slot[s].append((s,e,w))

    # Walk through sentence and emit either plain text or toggle button widgets
    items = []
    i = 0
    while i < len(sentence_text):
        if i < len(slot) and slot[i]:
            # If multiple words start at same char, emit all (rare)
            for (s,e,w) in slot[i]:
                btn = W.ToggleButton(description=w, layout=W.Layout(height="auto"))
                btn.observe(on_toggle_factory(w), names="value")
                items.append(btn)
                i = max(i, e)
        else:
            # accumulate contiguous plain segment
            j = i
            while j < len(sentence_text) and (j >= len(slot) or not slot[j]):
                j += 1
            plain = sentence_text[i:j]
            items.append(W.HTML(f"<span>{plain}</span>"))
            i = j
    return W.HBox(items, layout=W.Layout(flex_flow="row wrap"))

# ---------- selection state ----------
class ConcordanceSelectionState:
    def __init__(self):
        self.selected_terms = set()
        self.match_ids      = set()
        self.matched_rows   = []   # list of dicts for convenience

# ---------- main widget ----------
def launch_inline_overlay_widget(production_output: Dict[str, Any],
                                 db_path: str,
                                 default_sentence_id: Optional[int] = None,
                                 index_limit: int = 20000,
                                 on_selection: Optional[Callable[[List[str], List[Dict[str,Any]]], None]] = None):
    """
    Launch the overlay UI. If `on_selection` is provided, it will be called with
    (terms: List[str], matches: List[dict]) every time the selection changes.
    """
    sent_records, inv = load_sentences_and_index(db_path, limit=index_limit)

    # choose sentence
    sid_options = [s["sentence_id"] for s in production_output["sentence_analyses"]]
    if not sid_options:
        raise ValueError("No sentence_analyses present in production_output")
    if default_sentence_id is None:
        default_sentence_id = sid_options[0]

    dd_sid = W.Dropdown(options=sid_options, value=default_sentence_id, description="sentence_id:", layout=W.Layout(width="260px"))
    mode = W.ToggleButtons(options=["AND","OR"], value="AND", description="Query:", layout=W.Layout(width="220px"))
    out_sentence = W.Output()
    out_results = W.Output()
    state = ConcordanceSelectionState()

    # Build span-word positions for a given production sentence (word-level ranges in the sentence)
    word_pat = re.compile(r"\w[\w-]*", flags=re.UNICODE)
    def span_word_positions_for_sentence(prod, sentence_id):
        srec = next((s for s in prod["sentence_analyses"] if s["sentence_id"] == sentence_id), None)
        if not srec:
            return [], ""
        s_text = srec.get("sentence_text","")
        # collect intervals for spans (local sentence coords)
        intervals = []
        for sp in srec.get("span_analysis", []) or []:
            # sp may use absolute (global) coords; convert to local
            g0, g1 = sp.get("coords") or (sp.get("start_char"), sp.get("end_char"))
            if g0 is None or g1 is None: 
                continue
            s0 = srec.get("doc_start")
            if s0 is None:
                continue
            local0 = max(0, g0 - s0)
            local1 = max(0, g1 - s0)
            intervals.append((local0, local1))
        # find words that intersect any interval
        pos = []
        for m in word_pat.finditer(s_text):
            w0, w1 = m.span()
            if any(not (w1 <= a or w0 >= b) for (a,b) in intervals):
                pos.append((w0, w1, m.group(0)))
        return pos, s_text

    def combined_sentence_ids(terms, mode_):
        if not terms:
            return set()
        sets = [inv.get(t.lower(), set()) for t in terms]
        if not sets:
            return set()
        return set.intersection(*sets) if mode_ == "AND" else set.union(*sets)

    def render_results():
        out_results.clear_output()
        terms = sorted(state.selected_terms, key=str.lower)
        ids = combined_sentence_ids(terms, mode.value)
        state.match_ids = ids
        state.matched_rows = []
        html_chunks = []
        html_chunks.append(f"<div style='margin:4px 0;color:#666;'>Matches: {len(ids)} sentence(s) | Mode: <b>{mode.value}</b> | Terms: {', '.join(terms) if terms else '—'}</div>")
        if not ids:
            html_chunks.append("<em>Select buttons in the sentence above to query.</em>")
        else:
            shown = 0
            for sid in list(ids)[:200]:
                rec = sent_records.get(sid)
                if not rec: 
                    continue
                state.matched_rows.append(dict(sid=sid, **rec))
                s = bold_all_terms(rec["text"], terms)
                html_chunks.append(f"""<div style="margin:6px 0;">
                    <code style="font-size:13px;">{s}</code>
                    <div style="color:#666;font-size:12px;">{os.path.basename(rec['path'])} [{rec['start']}:{rec['end']}] (sid={sid})</div>
                </div>""")
                shown += 1
            if len(ids) > shown:
                html_chunks.append(f"<div style='color:#666;font-size:12px;'>…and {len(ids)-shown} more not shown.</div>")
        with out_results:
            display(HTML("".join(html_chunks)))
        if on_selection:
            try:
                on_selection(terms, state.matched_rows)
            except Exception as e:
                with out_results:
                    display(HTML(f"<div style='color:#b00;'>Selection callback error: {e}</div>"))

    def build_sentence_row(sid):
        out_sentence.clear_output()
        pos, s_text = span_word_positions_for_sentence(production_output, sid)

        def on_toggle_factory(word):
            def _obs(change):
                if change["name"] == "value":
                    if change["new"]:
                        state.selected_terms.add(word)
                    else:
                        state.selected_terms.discard(word)
                    render_results()
            return _obs

        if not s_text:
            with out_sentence: display(HTML("<em>Sentence text unavailable.</em>")); return
        if not pos:
            with out_sentence:
                display(HTML(f"<div style='margin-bottom:8px;'><b>Sentence:</b> {s_text}</div><em>No span words available for selection.</em>"))
                return

        # Build inline overlay
        inline = make_inline_sentence_buttons(s_text, pos, on_toggle_factory)
        with out_sentence:
            display(HTML("<div style='margin-bottom:6px;color:#444;'>Click one or more <b>highlighted words</b> to build a combined query.</div>"))
            display(inline)

    def on_sid_change(change):
        if change["name"] == "value":
            state.selected_terms.clear()
            out_results.clear_output()
            build_sentence_row(change["new"])
            render_results()

    def on_mode_change(change):
        if change["name"] == "value":
            render_results()

    dd_sid.observe(on_sid_change, names="value")
    mode.observe(on_mode_change, names="value")

    # initial render
    build_sentence_row(default_sentence_id)
    render_results()

    ui = W.VBox([
        W.HBox([dd_sid, mode]),
        W.HTML("<hr style='margin:6px 0;'>"),
        out_sentence,
        W.HTML("<hr style='margin:6px 0;'>"),
        W.HTML("<b>Concordances (combined query)</b>"),
        out_results
    ])
    display(ui)
    return ui, state
