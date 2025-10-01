# -*- coding: utf-8 -*-
# === Inline span buttons + combined AND/OR concordance over FlexiConc (SQL index) + SciCo graph (non-blocking) ===
import sqlite3, re, os, threading
from pathlib import Path
from collections import defaultdict
import ipywidgets as W
from IPython.display import display, HTML
from typing import List, Dict, Any

# SciCo + clustering + communities + summaries
from scico_graph_pipeline import build_graph_from_selection, ScicoConfig

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
    terms_sorted = sorted(set(t.lower() for t in terms), key=len, reverse=True)
    def repl(m): return f"<b>{m.group(0)}</b>"
    out = s
    for t in terms_sorted:
        out = re.sub(rf"(?i)\b{re.escape(t)}\b", repl, out)
    return out

def make_inline_sentence_buttons(sentence_text, span_word_positions, on_toggle_factory):
    slot = [[] for _ in range(len(sentence_text)+1)]
    for s,e,w in span_word_positions:
        s = max(0, min(s, len(sentence_text)))
        e = max(0, min(e, len(sentence_text)))
        slot[s].append((s,e,w))

    items, i = [], 0
    while i < len(sentence_text):
        if i < len(slot) and slot[i]:
            for (s,e,w) in slot[i]:
                btn = W.ToggleButton(description=w, layout=W.Layout(height="auto"))
                btn.observe(on_toggle_factory(w), names="value")
                items.append(btn)
                i = max(i, e)
        else:
            j = i
            while j < len(sentence_text) and (j >= len(slot) or not slot[j]):
                j += 1
            items.append(W.HTML(f"<span>{sentence_text[i:j]}</span>"))
            i = j
    return W.HBox(items, layout=W.Layout(flex_flow="row wrap"))

# ---------- main widget ----------
def launch_inline_overlay_widget(
    production_output,
    db_path: str,
    default_sentence_id: int = None,
    index_limit=20000,
    *,
    scico_mode: str = "auto",            # "auto" = run on each change, "manual" = click button
    scico_params: dict | None = None,    # passed into build_graph_from_selection
    on_graph=None                         # optional callback: on_graph(G, meta, rows, terms)
):
    """
    Concordance UI + SciCo graph (background).
    """

    # --- load flexiconc slice ---
    sent_records, inv = load_sentences_and_index(db_path, limit=index_limit)

    # choose sentence
    sid_options = [s["sentence_id"] for s in production_output["sentence_analyses"]]
    if default_sentence_id is None:
        default_sentence_id = sid_options[0] if sid_options else 0

    # === Top controls ===
    dd_sid   = W.Dropdown(options=sid_options, value=default_sentence_id,
                          description="sentence_id:", layout=W.Layout(width="260px"))
    mode     = W.ToggleButtons(options=["AND","OR"], value="AND", description="Query:", layout=W.Layout(width="220px"))
    use_vector = W.Checkbox(value=True, description="Vector search", layout=W.Layout(width="160px"))
    vector_backend = W.Text(value="mpnet", description="Embedding:", layout=W.Layout(width="260px"))
    use_faiss_query = W.Checkbox(value=True, description="Use FAISS", layout=W.Layout(width="140px"))

    # --- shortlist controls ---
    use_short = W.Checkbox(value=False, description="Use coherence shortlist")
    faiss_topk = W.IntSlider(value=32, min=5, max=200, step=1, description="FAISS top-k", layout=W.Layout(width="300px"))
    nprobe     = W.IntSlider(value=8, min=1, max=64, description="nprobe", layout=W.Layout(width="300px"))
    add_lsh    = W.Checkbox(value=True, description="Add MinHash LSH")
    lsh_thr    = W.FloatSlider(value=0.8, min=0.1, max=0.95, step=0.01, readout_format=".2f", description="LSH threshold")
    minhash_k  = W.IntSlider(value=5, min=2, max=10, description="MinHash k")
    cheap_len  = W.FloatSlider(value=0.25, min=0.0, max=1.0, step=0.05, description="Min len ratio")
    cheap_jac  = W.FloatSlider(value=0.08, min=0.0, max=1.0, step=0.01, description="Min Jaccard")
    use_coh    = W.Checkbox(value=False, description="Use SGNLP coherence")
    coh_thr    = W.FloatSlider(value=0.55, min=0.0, max=1.0, step=0.01, description="Coherence thr")

    # --- clustering controls ---
    clustering = W.Dropdown(options=["auto","kmeans","torque","both","none"], value="auto", description="Clustering:")
    kmeans_k   = W.IntSlider(value=5, min=2, max=50, description="kmeans_k")
    use_torque = W.Checkbox(value=False, description="Use Torque (auto)")

    # --- community controls ---
    community_on = W.Dropdown(options=[("All edges","all"),("Corefer only","corefer"),("Parent/Child","parent_child")],
                              value="all", description="Comm edges:")
    community_method = W.Dropdown(options=[("Greedy","greedy"),("Louvain","louvain"),("Leiden","leiden"),("LabelProp","labelprop"),("None","none")],
                                  value="greedy", description="Method:")

    # --- summarization controls ---
    summarize   = W.Checkbox(value=False, description="Summarize groups")
    summarize_on= W.Dropdown(options=[("Communities","community"),("KMeans","kmeans"),("Torque","torque")],
                             value="community", description="Summarize on:")
    methods     = W.SelectMultiple(options=[("Centroid rep","centroid"),("XSum (abstractive)","xsum"),("PreSumm top","presumm")],
                                   value=("centroid",), description="Methods:", rows=3, layout=W.Layout(width="280px"))
    num_sent    = W.IntSlider(value=1, min=1, max=5, description="XSum sents")
    sdg_topk    = W.IntSlider(value=3, min=1, max=10, description="SDG top-k")
    ce_model    = W.Text(value="cross-encoder/ms-marco-MiniLM-L6-v2", description="CE model:", layout=W.Layout(width="420px"))

    # --- sparsify / threshold controls ---
    prob_thr    = W.FloatSlider(value=0.55, min=0.0, max=1.0, step=0.01, description="SciCO prob≥")
    max_deg     = W.IntSlider(value=30, min=5, max=200, description="Max degree")
    keep_top    = W.IntSlider(value=30, min=5, max=200, description="Keep top-E")

    # --- outputs ---
    out_sent   = W.Output()
    out_res    = W.Output()
    out_scico  = W.Output()
    btn_scico  = W.Button(description="Run SciCo", button_style="success", icon="play")
    btn_scico.layout.display = "none" if scico_mode == "auto" else ""

    # --- summaries explorer UI ---
    out_summ = W.Output()
    box_summ = W.Accordion(children=[out_summ])
    box_summ.set_title(0, "Group summaries (XSum | PreSumm | Centroid rep | Centroid shortlist)")

    def _fmt_sdg(block):
        if not block: return ""
        rows = []
        for item in block:
            g = item.get("goal")
            s = item.get("score")
            if g is None or s is None: 
                continue
            rows.append(f"<div style='font-size:12px;color:#333;'>{float(s):.4f} — <code>{g}</code></div>")
        return "".join(rows)

    def _render_group_summary(meta, rows, cid):
        sm = (meta.get("summaries") or {}).get(cid, {})
        if not sm:
            return "<em>No summary for this group.</em>"

        parts = []

        # XSum
        if "xsum_summary" in sm:
            xsum = sm["xsum_summary"]
            xsum_sdg = _fmt_sdg(sm.get("xsum_sdg"))
            parts.append(
                "<div><b>XSum (1-sent):</b><br>"
                f"<div style='margin:3px 0 6px 8px'>{xsum}</div>{xsum_sdg}</div>"
            )

        # PreSumm
        if "presumm_top_sent" in sm:
            ps = sm["presumm_top_sent"]; sc = sm.get("presumm_top_score")
            presumm_sdg = _fmt_sdg(sm.get("presumm_sdg"))
            parts.append(
                "<div><b>PreSumm top:</b><br>"
                f"<div style='margin:3px 0 6px 8px'>{ps}"
                + (f" <span style='color:#666'>(score: {float(sc):.4f})</span>" if isinstance(sc, (int,float)) else "")
                + "</div>" + presumm_sdg + "</div>"
            )

        # Centroid representative
        if "representative" in sm:
            rep = sm["representative"]
            rep_sdg = _fmt_sdg(sm.get("representative_sdg"))
            parts.append(
                "<div><b>Centroid representative:</b><br>"
                f"<div style='margin:3px 0 6px 8px'>{rep}</div>{rep_sdg}</div>"
            )

        # Centroid shortlist (kept)
        kept = sm.get("centroid_kept")
        if kept:
            thr = sm.get("centroid_threshold")
            head = "<div><b>Centroid shortlist</b>"
            if isinstance(thr, (int,float)): head += f" <span style='color:#666'>(sim ≥ {thr:.2f})</span>"
            head += "</div>"
            items = "".join(
                f"<div style='margin:2px 0 2px 8px'>"
                f"<span style='color:#999'>{i:02d}</span> — "
                f"<span style='color:#555'>{float(sim):.3f}</span> — "
                f"{sent}</div>"
                for i, item in enumerate(kept, 1)
                for sent, sim in [(item.get('sentence',''), item.get('sim', 0.0))]
            )
            parts.append(head + items)

        if not parts:
            return "<em>No summary items produced (check summary_methods).</em>"

        return "<div style='line-height:1.35em'>" + "".join(parts) + "</div>"

    def update_summary_panel(meta, rows):
        """Populate the group summaries panel after each SciCo run."""
        out_summ.clear_output()
        summaries = meta.get("summaries") or {}
        if not summaries:
            with out_summ:
                display(HTML("<div style='color:#666'>No summaries (enable summarize=True and summary_methods in scico_params).</div>"))
            return

        cids = sorted(summaries.keys(), key=lambda x: (isinstance(x, int), x))
        dd = W.Dropdown(options=cids, description="Group:", layout=W.Layout(width="240px"))
        area = W.Output()

        def _render(_):
            area.clear_output()
            cid = dd.value
            if cid is None: return
            html = _render_group_summary(meta, rows, cid)
            with area:
                display(HTML(html))

        dd.observe(_render, names="value")
        _render(None)  # initial

        with out_summ:
            display(W.VBox([dd, area]))

    sel = set()

    # --- span positions in the chosen production sentence ---
    word_pat = re.compile(r"\w[\w-]*", flags=re.UNICODE)
    def span_word_positions_for_sentence(prod, sentence_id):
        srec = next((s for s in prod["sentence_analyses"] if s["sentence_id"] == sentence_id), None)
        if not srec:
            return [], ""
        s_text = srec.get("sentence_text","")
        intervals = []
        for sp in srec.get("span_analysis", []) or []:
            g0, g1 = sp.get("coords", (None, None))
            s0 = srec.get("doc_start")
            if s0 is not None and g0 is not None and g1 is not None:
                intervals.append((max(0, g0 - s0), max(0, g1 - s0)))
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

    # --- background SciCo runner ---
    current_rows: List[Dict[str, Any]] = []
    current_embeddings = None
    current_meta: Dict[str, Any] = {}
      
    
    
    def run_scico_async(terms, rows, embeddings):
        if not terms or not rows:
            out_scico.clear_output(); return    
        # Build argument dict for pipeline (forward all knobs; defaults come from UI here)
        params = dict(
            selected_terms=terms,
            sdg_targets=None,           # plug your SDG dict if you want sentence-level CE features
            kmeans_k=int(kmeans_k.value),
            use_torque=bool(use_torque.value),
            scico_cfg=ScicoConfig(prob_threshold=float(prob_thr.value)),
            embedder=None,              # pass your SentenceTransformer if you want
            add_layout=True,

            # Use pipeline's internal shortlist path when requested
            candidate_pairs=None,
            use_coherence_shortlist=bool(use_short.value),
            coherence_opts=dict(
                faiss_topk=int(faiss_topk.value),
                nprobe=int(nprobe.value),
                add_lsh=bool(add_lsh.value),
                lsh_threshold=float(lsh_thr.value),
                minhash_k=int(minhash_k.value),
                cheap_len_ratio=float(cheap_len.value),
                cheap_jaccard=float(cheap_jac.value),
                use_coherence=bool(use_coh.value),
                coherence_threshold=float(coh_thr.value),
                max_pairs=None,
            ),

            # clustering / communities
            clustering_method=clustering.value,
            community_on=community_on.value,
            community_method=community_method.value,
            community_weight="prob",

            # sparsify
            max_degree=int(max_deg.value),
            top_edges_per_node=int(keep_top.value),

            # summaries
            summarize=bool(summarize.value),
            summarize_on=summarize_on.value,
            summary_methods=list(methods.value),
            summary_opts=dict(
                num_sentences=int(num_sent.value),     # for XSum
                # presumm_model=ext_model,             # <- plug if available
                # presumm_tokenizer=presumm_tokenizer, # <- plug if available
                # device="cuda:0",
                sdg_targets=None,                       # supply to re-rank summaries vs SDG goals
                sdg_top_k=int(sdg_topk.value),
                cross_encoder_model=ce_model.value if ce_model.value.strip() else None,

                # centroid thresholding (new add-on)
                centroid_sim_threshold=0.55,
                centroid_top_n=5,
                centroid_store_vector=False,
            ),
        )
        
        if embeddings is not None:
            params["precomputed_embeddings"] = embeddings

        # allow caller overrides through scico_params
        if scico_params:
            params.update(scico_params)

        def worker():
            try:
                G, meta = build_graph_from_selection(rows, **params)
                out_scico.clear_output()
                with out_scico:
                    comm = meta.get('communities', {})
                    num_comm = None
                    if isinstance(comm, dict):
                        try:
                            num_comm = len(set(comm.values()))
                        except Exception:
                            num_comm = None
                    msg = (f"<div style='color:#3a6;'>SciCo graph ready: "
                           f"|V|={G.number_of_nodes()} |E|={G.number_of_edges()} "
                           f"|pairs_scored|={meta.get('pairs_scored')} "
                           f"{f'|communities|={num_comm}' if num_comm is not None else ''}"
                           f"</div>")
                    display(HTML(msg))
                    # If summaries were requested, show a quick peek
                    if params.get("summarize") and meta.get("summaries"):
                        display(HTML("<b>Summaries (peek):</b>"))
                        shown = 0
                        for cid, sm in list(meta["summaries"].items()):
                            parts = []
                            if "representative" in sm:
                                parts.append(f"<i>rep</i>: {sm['representative']}")
                            if "xsum_summary" in sm:
                                parts.append(f"<i>xsum</i>: {sm['xsum_summary']}")
                            if "presumm_top_sent" in sm:
                                parts.append(f"<i>presumm</i>: {sm['presumm_top_sent']} ({sm.get('presumm_top_score','')})")
                            if parts:
                                display(HTML(f"<div style='margin:4px 0;'><b>Group {cid}</b>: " + " | ".join(parts) + "</div>"))
                            shown += 1
                            if shown >= 6:
                                display(HTML("<div style='color:#666;font-size:12px;'>…(truncated)</div>"))
                                break

                # Populate the rich summaries panel
                update_summary_panel(meta, rows)

                if callable(on_graph):
                    on_graph(G, meta, rows, terms)
            except Exception as e:
                out_scico.clear_output()
                with out_scico:
                    display(HTML(f"<div style='color:#b00;'>SciCo error: {e}</div>"))

        out_scico.clear_output()
        with out_scico:
            display(HTML(f"<div style='color:#777;'>Running SciCo on {len(rows)} sentences…</div>"))
        threading.Thread(target=worker, daemon=True).start()

    # --- render concordance list + (optionally) kick SciCo ---
    def render_results():
        nonlocal current_rows, current_embeddings, current_meta
        out_res.clear_output()
        terms = sorted(sel, key=str.lower)
        

        html = []
        
        if not terms:
            html.append("<em>Select buttons in the sentence above to query.</em>")
        else:
            vector_mode = bool(use_vector.value and vector_backend.value.strip())
            if vector_mode:
                try:
                    from flexiconc_adapter import query_concordance

                    res = query_concordance(
                        db_path,
                        terms,
                        mode=mode.value,
                        limit=200,
                        vector_backend=vector_backend.value.strip(),
                        use_faiss=bool(use_faiss_query.value),
                    )
                    current_rows = res.get("rows") or []
                    current_embeddings = res.get("embeddings")
                    current_meta = res.get("meta") or {}
                    html.append(
                        f"<div style='margin:4px 0;color:#666;'>Vector matches: {len(current_rows)} | Backend: "
                        f"{current_meta.get('vector_backend','?')} | Mode: <b>{current_meta.get('mode','vector')}</b></div>"
                    )
                    for row in current_rows[:200]:
                        txt = row.get("text", "")
                        score = row.get("score")
                        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else ""
                        html.append(
                            f"<div style='margin:6px 0;'>"
                            f"<code style=\"font-size:13px;\">{bold_all_terms(txt, terms)}</code>"
                            f"<div style='color:#666;font-size:12px;'>"
                            f"{os.path.basename(row.get('path',''))}"
                            f" [{row.get('start','?')}:{row.get('end','?')}]"
                            f" (doc={row.get('doc_id','?')} sid={row.get('sentence_id','?')} score={score_str})"
                            f"</div></div>"
                        )
                except Exception as exc:
                    html.append(f"<div style='color:#b00;'>Vector search failed: {exc}</div>")
                    vector_mode = False

            if not current_rows:
                ids = combined_sentence_ids(terms, mode.value)
                html.append(
                    f"<div style='margin:4px 0;color:#666;'>Matches: {len(ids)} sentence(s) | Mode: <b>{mode.value}</b>"
                    f" | Terms: {', '.join(terms)}</div>"
                )
                if not ids:
                    html.append("<em>No lexical matches.</em>")
                else:
                    shown = 0
                    rows_local = []
                    for sid in list(ids)[:200]:
                        rec = sent_records.get(sid)
                        if not rec:
                            continue
                        rows_local.append(dict(
                            text=rec["text"],
                            path=rec["path"],
                            start=rec["start"],
                            end=rec["end"],
                            doc_id=os.path.basename(rec["path"]),
                            sentence_id=sid,
                        ))
                        s = bold_all_terms(rec['text'], terms)
                        html.append(
                            f"<div style='margin:6px 0;'><code style=\"font-size:13px;\">{s}</code>"
                            f"<div style='color:#666;font-size:12px;'>"
                            f"{os.path.basename(rec['path'])} [{rec['start']}:{rec['end']}] (sid={sid})"
                            f"</div></div>"
                        )
                        shown += 1
                    if len(ids) > shown:
                        html.append(f"<div style='color:#666;font-size:12px;'>…and {len(ids)-shown} more not shown.</div>")
                    current_rows = rows_local
        with out_res:
            display(HTML("".join(html)))

        if scico_mode == "auto":
            run_scico_async(terms, ids)
            run_scico_async(terms, current_rows, current_embeddings)

        btn_scico.disabled = not bool(terms and ids)
        btn_scico.disabled = not bool(terms and current_rows)

    # --- UI building helpers ---
    def build_sentence_row(sid):
        out_sent.clear_output()
        pos, s_text = span_word_positions_for_sentence(production_output, sid)

        def on_toggle_factory(word):
            def _obs(change):
                if change["name"] == "value":
                    if change["new"]:
                        sel.add(word)
                    else:
                        sel.discard(word)
                    render_results()
            return _obs

        if not s_text:
            with out_sent: display(HTML("<em>Sentence text unavailable.</em>")); return
        if not pos:
            with out_sent:
                display(HTML(f"<div style='margin-bottom:8px;'><b>Sentence:</b> {s_text}</div><em>No span words available for selection.</em>"))
                return

        inline = make_inline_sentence_buttons(s_text, pos, on_toggle_factory)
        with out_sent:
            display(HTML("<div style='margin-bottom:6px;color:#444;'>Click one or more <b>highlighted words</b> to build a combined query.</div>"))
            display(inline)

    def on_sid_change(change):
        if change["name"] == "value":
            sel.clear()
            out_res.clear_output()
            out_scico.clear_output()
            out_summ.clear_output()
            build_sentence_row(change["new"])
            render_results()

    def on_mode_change(change):
        if change["name"] == "value":
            render_results()

    def on_scico_click(_):
        terms = sorted(sel, key=str.lower)
        ids = combined_sentence_ids(terms, mode.value)
        run_scico_async(terms, ids)

    dd_sid.observe(on_sid_change, names="value")
    mode.observe(on_mode_change, names="value")
    use_vector.observe(lambda change: render_results() if change["name"] == "value" else None, names="value")
    use_faiss_query.observe(lambda change: render_results() if change["name"] == "value" else None, names="value")
    vector_backend.observe(lambda change: render_results() if change["name"] == "value" else None, names="value")
    btn_scico.on_click(on_scico_click)

    # initial render
    build_sentence_row(default_sentence_id)
    render_results()

    # --- layout ---
    shortlist_box = W.Accordion(children=[W.VBox([
        use_short, W.HBox([faiss_topk, nprobe]),
        W.HBox([add_lsh, lsh_thr, minhash_k]),
        W.HBox([cheap_len, cheap_jac]),
        W.HBox([use_coh, coh_thr]),
    ])])
    shortlist_box.set_title(0, "Candidate shortlist (FAISS / LSH / Coherence)")

    cluster_box = W.Accordion(children=[W.VBox([
        W.HBox([clustering, kmeans_k, use_torque])
    ])])
    cluster_box.set_title(0, "Embedding clustering")

    community_box = W.Accordion(children=[W.VBox([
        W.HBox([community_on, community_method])
    ])])
    community_box.set_title(0, "Community detection")

    summary_box = W.Accordion(children=[W.VBox([
        W.HBox([summarize, summarize_on]),
        W.HBox([methods, W.VBox([num_sent, sdg_topk, ce_model])]),
    ])])
    summary_box.set_title(0, "Summaries (centroid / XSum / PreSumm + SDG re-rank)")

    sparsify_box = W.Accordion(children=[W.VBox([
        W.HBox([prob_thr]),
        W.HBox([max_deg, keep_top]),
    ])])
    sparsify_box.set_title(0, "Edge threshold & sparsify")

    header = [dd_sid, mode]
    if scico_mode == "manual":
        header.append(btn_scico)
    
    vector_row = W.HBox([use_vector, vector_backend, use_faiss_query])

    ui = W.VBox([
        W.HBox(header),
        vector_row,
        W.HTML("<hr style='margin:6px 0;'>"),
        out_sent,
        W.HTML("<hr style='margin:6px 0;'>"),
        W.HTML("<b>Concordances (combined query)</b>"),
        out_res,
        W.HTML("<hr style='margin:6px 0;'>"),
        shortlist_box,
        cluster_box,
        community_box,
        summary_box,
        sparsify_box,
        W.HTML("<hr style='margin:6px 0;'>"),
        out_scico,   # status line
        box_summ,    # group summaries explorer
    ])
    display(ui)
