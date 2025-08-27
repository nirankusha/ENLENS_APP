# -*- coding: utf-8 -*-
"""
flexiconc_adapter.py
--------------------
Adapter to persist/reload the unified `production_output` into a FlexiConc-style
SQLite database (per-sentence rows), with optional sentence embeddings.

Tables (created if missing):
- documents(doc_id TEXT PRIMARY KEY, uri TEXT, created_at DATETIME, full_text TEXT)
- sentences(doc_id TEXT, sentence_id INT, start INT, end INT, text TEXT,
            label INT, confidence REAL, consensus TEXT,
            token_analysis_json TEXT, span_analysis_json TEXT,
            PRIMARY KEY(doc_id, sentence_id))
- chains(doc_id TEXT, chain_id INT, representative TEXT, mentions_json TEXT,
         PRIMARY KEY(doc_id, chain_id))
- clusters(doc_id TEXT, cluster_id INT, members_json TEXT, PRIMARY KEY(doc_id, cluster_id))
- embeddings(doc_id TEXT, sentence_id INT, model TEXT, dim INT, vector BLOB,
             PRIMARY KEY(doc_id, sentence_id, model))

You can change/extend this schema if your existing FlexiConc has different names;
just adjust the mapping functions below.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Iterable
import sqlite3, json, os, datetime
import contextlib

# If you already have an embedder in helper.py, we’ll use it; else lazy fallback/no-op.
try:
    from helper import get_sentence_embedding  # optional: (text) -> np.ndarray[float32]
except Exception:
    get_sentence_embedding = None

def _ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents(
        doc_id TEXT PRIMARY KEY,
        uri TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        full_text TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sentences(
        doc_id TEXT,
        sentence_id INTEGER,
        start INTEGER,
        end INTEGER,
        text TEXT,
        label INTEGER,
        confidence REAL,
        consensus TEXT,
        token_analysis_json TEXT,
        span_analysis_json TEXT,
        PRIMARY KEY(doc_id, sentence_id)
    )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sent_doc ON sentences(doc_id)")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chains(
        doc_id TEXT,
        chain_id INTEGER,
        representative TEXT,
        mentions_json TEXT,
        PRIMARY KEY(doc_id, chain_id)
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS clusters(
        doc_id TEXT,
        cluster_id INTEGER,
        members_json TEXT,
        PRIMARY KEY(doc_id, cluster_id)
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings(
        doc_id TEXT,
        sentence_id INTEGER,
        model TEXT,
        dim INTEGER,
        vector BLOB,
        PRIMARY KEY(doc_id, sentence_id, model)
    )""")
    conn.commit()

def _to_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))

def open_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)  # autocommit
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA mmap_size=268435456;")  # 256MB
    conn.commit()
    _ensure_schema(conn)
    return conn

def upsert_document(conn: sqlite3.Connection, doc_id: str, full_text: str, uri: Optional[str] = None):
    conn.execute(
        "INSERT INTO documents(doc_id, uri, full_text) VALUES (?,?,?) "
        "ON CONFLICT(doc_id) DO UPDATE SET uri=excluded.uri, full_text=excluded.full_text",
        (doc_id, uri, full_text),
    )
    conn.commit()

def upsert_sentences(conn: sqlite3.Connection, doc_id: str, production_output: Dict[str, Any]):
    sents = production_output.get("sentence_analyses", []) or []
    rows = []
    for sa in sents:
        sid = int(sa.get("sentence_id", 0))
        st  = int(sa.get("doc_start", 0))
        en  = int(sa.get("doc_end", 0))
        txt = sa.get("sentence_text", "")
        cls = sa.get("classification", {}) or {}
        label = int(cls.get("label", -1)) if cls.get("label") is not None else None
        conf  = float(cls.get("confidence", 0.0)) if cls.get("confidence") is not None else None
        cons  = cls.get("consensus", None)
        tkj   = _to_json(sa.get("token_analysis") or {})
        spj   = _to_json(sa.get("span_analysis") or [])
        rows.append((doc_id, sid, st, en, txt, label, conf, cons, tkj, spj))
    conn.executemany("""
        INSERT INTO sentences(doc_id, sentence_id, start, end, text, label, confidence, consensus,
                              token_analysis_json, span_analysis_json)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(doc_id, sentence_id) DO UPDATE SET
            start=excluded.start, end=excluded.end, text=excluded.text,
            label=excluded.label, confidence=excluded.confidence, consensus=excluded.consensus,
            token_analysis_json=excluded.token_analysis_json, span_analysis_json=excluded.span_analysis_json
    """, rows)
    conn.commit()

def upsert_chains(conn: sqlite3.Connection, doc_id: str, production_output: Dict[str, Any]):
    ca = (production_output.get("coreference_analysis") or {})
    chains = ca.get("chains", []) or []
    if not chains:
        return
    rows = []
    for ch in chains:
        cid = int(ch.get("chain_id", -1))
        rep = ch.get("representative", None)
        mentions_json = _to_json(ch.get("mentions") or [])
        rows.append((doc_id, cid, rep, mentions_json))
    conn.executemany("""
        INSERT INTO chains(doc_id, chain_id, representative, mentions_json)
        VALUES (?,?,?,?)
        ON CONFLICT(doc_id, chain_id) DO UPDATE SET
            representative=excluded.representative, mentions_json=excluded.mentions_json
    """, rows)
    conn.commit()

def upsert_clusters(conn: sqlite3.Connection, doc_id: str, production_output: Dict[str, Any]):
    cl = (production_output.get("cluster_analysis") or {})
    clusters = cl.get("clusters", []) or []
    if not clusters:
        return
    rows = [(doc_id, i, _to_json(members)) for i, members in enumerate(clusters)]
    conn.executemany("""
        INSERT INTO clusters(doc_id, cluster_id, members_json)
        VALUES (?,?,?)
        ON CONFLICT(doc_id, cluster_id) DO UPDATE SET
            members_json=excluded.members_json
    """, rows)
    conn.commit()

def _np_to_blob(vec) -> Optional[bytes]:
    try:
        import numpy as np
        if vec is None: return None
        v = np.asarray(vec, dtype=np.float32)
        return v.tobytes(order="C")
    except Exception:
        return None

def upsert_sentence_embeddings(conn: sqlite3.Connection, doc_id: str,
                               sentences: List[str],
                               model_name: str = "helper.get_sentence_embedding"):
    if get_sentence_embedding is None:
        return  # no-op if you didn’t expose an embedder
    import numpy as np
    rows = []
    for sid, text in enumerate(sentences):
        vec = get_sentence_embedding(text)  # expected np.ndarray[float32] shape (d,)
        if vec is None: 
            continue
        blob = _np_to_blob(vec)
        rows.append((doc_id, sid, model_name, int(vec.shape[-1]), blob))
    if rows:
        conn.executemany("""
            INSERT INTO embeddings(doc_id, sentence_id, model, dim, vector)
            VALUES (?,?,?,?,?)
            ON CONFLICT(doc_id, sentence_id, model) DO UPDATE SET
                dim=excluded.dim, vector=excluded.vector
        """, rows)
        conn.commit()

def get_sentence_embedding_cached(conn, doc_id, sentence_id, text, model_name="paraphrase-mpnet-base-v2"):
    """
    Returns np.ndarray[float32, dim] embedding for a sentence.
    Persists into 'embeddings' table to amortize cost.
    """
    cur = conn.cursor()
    try:
        cur.execute("SELECT vector, dim FROM embeddings WHERE doc_id=? AND sentence_id=? AND model=?",
                    (str(doc_id), int(sentence_id), model_name))
        row = cur.fetchone()
        if row:
            vec = np.frombuffer(row[0], dtype="float32")
            return vec
    except Exception:
        pass

    # Fallback: compute via helper.sim_model
    from helper import sim_model
    emb = sim_model.encode([text])[0].astype("float32")
    try:
        cur.execute(
            "INSERT OR REPLACE INTO embeddings(doc_id, sentence_id, model, dim, vector) VALUES (?,?,?,?,?)",
            (str(doc_id), int(sentence_id), model_name, int(emb.shape[0]), emb.tobytes())
        )
        conn.commit()
    except Exception:
        conn.rollback()
    return emb


def export_production_to_flexiconc(db_path: str, doc_id: str, production_output: Dict[str, Any],
                                   uri: Optional[str] = None, write_embeddings: bool = False):
    conn = open_db(db_path)
    try:
        upsert_document(conn, doc_id, production_output.get("full_text", ""), uri=uri)
        upsert_sentences(conn, doc_id, production_output)
        upsert_chains(conn, doc_id, production_output)
        upsert_clusters(conn, doc_id, production_output)
        if write_embeddings:
            sentences = [sa.get("sentence_text","") for sa in production_output.get("sentence_analyses",[])]
            upsert_sentence_embeddings(conn, doc_id, sentences)
    finally:
        conn.close()

def load_production_from_flexiconc(db_path: str, doc_id: str) -> Dict[str, Any]:
    conn = open_db(db_path)
    try:
        cur = conn.execute("SELECT full_text FROM documents WHERE doc_id=?", (doc_id,))
        row = cur.fetchone()
        full_text = row[0] if row else ""

        # sentences
        srows = conn.execute("""
           SELECT sentence_id, start, end, text, label, confidence, consensus,
                  token_analysis_json, span_analysis_json
           FROM sentences WHERE doc_id=? ORDER BY sentence_id
        """, (doc_id,)).fetchall()
        sentence_analyses = []
        for sid, st, en, txt, lab, conf, cons, tkj, spj in srows:
            tk = json.loads(tkj or "{}")
            sp = json.loads(spj or "[]")
            sentence_analyses.append({
                "sentence_id": sid,
                "sentence_text": txt,
                "doc_start": st, "doc_end": en,
                "classification": {"label": lab, "confidence": conf, "consensus": cons,
                                   "score": conf, "class_id": lab},
                "token_analysis": tk,
                "span_analysis": sp,
                "metadata": {}
            })

        # chains
        crows = conn.execute("""
          SELECT chain_id, representative, mentions_json FROM chains WHERE doc_id=? ORDER BY chain_id
        """, (doc_id,)).fetchall()
        chains = []
        for cid, rep, mjson in crows:
            chains.append({"chain_id": cid, "representative": rep, "mentions": json.loads(mjson or "[]")})

        # clusters
        clrows = conn.execute("""
          SELECT cluster_id, members_json FROM clusters WHERE doc_id=? ORDER BY cluster_id
        """, (doc_id,)).fetchall()
        clusters = [json.loads(members or "[]") for _, members in clrows]

        return {
            "full_text": full_text,
            "document_analysis": {},
            "coreference_analysis": {"num_chains": len(chains), "chains": chains},
            "sentence_analyses": sentence_analyses,
            "cluster_analysis": {"clusters": clusters, "clusters_dict": {str(i): c for i,c in enumerate(clusters)}, "graphs_json": {}}
        }
    finally:
        conn.close()

"""
Created on Sat Aug 16 19:08:03 2025

@author: niran
"""

