
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
import networkx as nx
import matplotlib.pyplot as plt
from helper import sim_model
from scico_graph_pipeline import build_graph_from_selection, ScicoConfig
from ann_index import FaissIndex, build_span_lsh
from coherence_sampler import shortlist_by_coherence

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
# add this import near the top of your cell
from helper import sim_model
# ---------- main widget ----------
def launch_inline_overlay_widget(
    production_output,
    db_path: str,
    default_sentence_id: int = None,
    index_limit=20000,
    *,
    scico_autorun: bool = True,
    scico_params: dict | None = None,
    on_graph=None,   # optional callback: (G, meta, rows, terms)
):
    """
    If scico_autorun=True, every selection change will:
      1) build a list of matched rows (text, path, start, end),
      2) run SciCo + clustering (build_graph_from_selection),
      3) call on_graph(G, meta, rows, terms) if provided.
    """
    sent_records, inv = load_sentences_and_index(db_path, limit=index_limit)

    # choose sentence
    sid_options = [s["sentence_id"] for s in production_output["sentence_analyses"]]
    if default_sentence_id is None:
        default_sentence_id = sid_options[0] if sid_options else 0

    dd_sid = W.Dropdown(options=sid_options, value=default_sentence_id, description="sentence_id:", layout=W.Layout(width="260px"))
    mode = W.ToggleButtons(options=["AND","OR"], value="AND", description="Query:", layout=W.Layout(width="220px"))
    out_sentence = W.Output()
    out_results = W.Output()
    out_scico   = W.Output()   # NEW: small status panel for SciCo/graph summary
    sel = set()

    # Build span-word positions for a given production sentence (word-level ranges in the sentence)
    word_pat = re.compile(r"\w[\w-]*", flags=re.UNICODE)
    def span_word_positions_for_sentence(prod, sentence_id):
        srec = next((s for s in prod["sentence_analyses"] if s["sentence_id"] == sentence_id), None)
        if not srec:
            return [], ""
        s_text = srec.get("sentence_text","")
        # collect intervals for spans (local sentence coords)
        intervals = []
        for sp in srec.get("span_analysis", []):
            # We assume sp["coords"] are global; compute local within this sentence
            g0, g1 = sp["coords"]
            s0 = srec["doc_start"] if "doc_start" in srec else None
            if s0 is not None:
                local0 = max(0, g0 - s0)
                local1 = max(0, g1 - s0)
                intervals.append((local0, local1))
        # find words that intersect any interval
        pos = []
        for m in word_pat.finditer(s_text):
            w0, w1 = m.span()
            # does (w0,w1) intersect any span interval?
            if any(not (w1 <= a or w0 >= b) for (a,b) in intervals):
                pos.append((w0, w1, m.group(0)))
        return pos, s_text

    # Build combined result set based on selection and mode
    def combined_sentence_ids(terms, mode_):
        if not terms:
            return set()
        sets = [inv.get(t.lower(), set()) for t in terms]
        if not sets:
            return set()
        return set.intersection(*sets) if mode_ == "AND" else set.union(*sets)

    # ---- SciCo runner (uses scico_graph_pipeline) ----
    
def run_scico(terms, ids):
    if not terms:
        out_scico.clear_output()
        return
    if ids is None or not ids:
        out_scico.clear_output()
        return

    rows = []
    for sid in ids:
        rec = sent_records.get(sid)
        if rec:
            rows.append(dict(sid=sid, text=rec['text'], path=rec['path'], start=rec['start'], end=rec['end']))
    if not rows:
        out_scico.clear_output()
        return

    maxN = (scico_params or {}).get('max_scico_sentences', None)
    if maxN and len(rows) > maxN:
        rows = rows[:maxN]

    raw = dict(
        selected_terms=terms,
        sdg_targets=(scico_params or {}).get('sdg_targets'),
        kmeans_k=(scico_params or {}).get('kmeans_k', 5),
        use_torque=(scico_params or {}).get('use_torque', False),
        scico_cfg=(scico_params or {}).get('scico_cfg', ScicoConfig(prob_threshold=0.55)),
        embedder=(scico_params or {}).get('embedder', None),
        add_layout=(scico_params or {}).get('add_layout', True),
        max_degree=(scico_params or {}).get('max_degree', 30),
        top_edges_per_node=(scico_params or {}).get('top_edges_per_node', 30),
    )
    ALLOWED = {'selected_terms','sdg_targets','kmeans_k','use_torque','scico_cfg','embedder','add_layout','max_degree','top_edges_per_node'}
    params = {k:v for k,v in raw.items() if k in ALLOWED and v is not None}

    G, meta = build_graph_from_selection(rows, **params)

    out_scico.clear_output()
    with out_scico:
        display(HTML(
            f"<div style='color:#3a6;'>SciCo graph updated: |V|={G.number_of_nodes()} |E|={G.number_of_edges()} (terms: {', '.join(terms)})</div>"
        ))
    if callable(on_graph):
        on_graph(G, meta, rows, terms)

def run_scico_with_pairs(terms, rows_tuples, candidate_pairs):
    # rows_tuples: [(sid, text, path, start, end), ...]
    # candidate_pairs: set[(i,j)] indices into rows_tuples
    if not scico_autorun or not terms or not rows_tuples:
        out_scico.clear_output()
        return
    rows = [dict(sid=sid, text=text, path=path, start=start, end=end) for (sid, text, path, start, end) in rows_tuples]
    local_params = dict(scico_params or {})
    local_params['candidate_pairs'] = list(candidate_pairs) if candidate_pairs else None

    raw = dict(
        selected_terms=terms,
        sdg_targets=local_params.get('sdg_targets'),
        kmeans_k=local_params.get('kmeans_k', 5),
        use_torque=local_params.get('use_torque', False),
        scico_cfg=local_params.get('scico_cfg', ScicoConfig(prob_threshold=0.55)),
        embedder=local_params.get('embedder', None),
        add_layout=local_params.get('add_layout', True),
        candidate_pairs=local_params.get('candidate_pairs'),
        max_degree=local_params.get('max_degree', 30),
        top_edges_per_node=local_params.get('top_edges_per_node', 30),
    )
    ALLOWED = {'selected_terms','sdg_targets','kmeans_k','use_torque','scico_cfg','embedder','add_layout','candidate_pairs','max_degree','top_edges_per_node'}
    params = {k:v for k,v in raw.items() if k in ALLOWED and v is not None}

    G, meta = build_graph_from_selection(rows, **params)
    out_scico.clear_output()
    with out_scico:
        display(HTML(
            f"<div style='color:#3a6;'>SciCo graph updated: |V|={G.number_of_nodes()} |E|={G.number_of_edges()} (terms: {', '.join(terms)})</div>"
        ))
    if callable(on_graph):
        on_graph(G, meta, rows, terms)
 