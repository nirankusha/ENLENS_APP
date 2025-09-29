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
import gzip, io
import numpy as np
try:
    import scipy.sparse as sp
except Exception:
    sp = None  # allow import even if scipy not present


# If you already have an embedder in helper.py, we’ll use it; else lazy fallback/no-op.
try:
    from helper import get_sentence_embedding  # optional: (text) -> np.ndarray[float32]
except Exception:
    get_sentence_embedding = None

def _ensure_schema(conn: sqlite3.Connection):
    cur = conn.cursor()

    # documents: keep columns minimal and types stable
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents(
        doc_id      TEXT PRIMARY KEY,
        uri         TEXT,
        created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
        text_length INTEGER,
        full_text   TEXT
    )""")

    # sentences: JSON stored as TEXT
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sentences(
        doc_id TEXT,
        sentence_id INTEGER,
        start INTEGER,
        end   INTEGER,
        text  TEXT,
        label INTEGER,
        confidence REAL,
        consensus TEXT,
        token_analysis_json TEXT,
        span_analysis_json  TEXT,
        PRIMARY KEY(doc_id, sentence_id)
    )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sent_doc ON sentences(doc_id)")

    # chains
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chains(
        doc_id TEXT,
        chain_id INTEGER,
        representative TEXT,
        mentions_json  TEXT,
        PRIMARY KEY(doc_id, chain_id)
    )""")

    # clusters (optional)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS clusters(
        doc_id TEXT,
        cluster_id INTEGER,
        members_json TEXT,
        PRIMARY KEY(doc_id, cluster_id)
    )""")

    # embeddings (optional)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS embeddings(
        doc_id TEXT,
        sentence_id INTEGER,
        model TEXT,
        dim INTEGER,
        vector BLOB,
        PRIMARY KEY(doc_id, sentence_id, model)
    )""")

    # indices payloads
    cur.execute("""
    CREATE TABLE IF NOT EXISTS indices(
        doc_id  TEXT,
        kind    TEXT,       -- 'trie' | 'cooc_vocab' | 'cooc_rows'
        payload BLOB,       -- gzipped JSON or npz bytes
        PRIMARY KEY(doc_id, kind)
    )""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_indices_kind ON indices(kind)")

    conn.commit()

def _migrate_documents_table(conn: sqlite3.Connection):
    cur = conn.cursor()
    cols = {r[1] for r in cur.execute("PRAGMA table_info(documents)").fetchall()}
    if "text_length" not in cols:
        cur.execute("ALTER TABLE documents ADD COLUMN text_length INTEGER")
    if "full_text" not in cols:
        cur.execute("ALTER TABLE documents ADD COLUMN full_text TEXT")
    conn.commit()

def _to_json(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))

def _blob_from_json(obj) -> bytes:
    raw = json.dumps(obj).encode("utf-8")
    return gzip.compress(raw)

def _json_from_blob(b: bytes):
    return json.loads(gzip.decompress(b).decode("utf-8"))

def _blob_from_npz(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()

def _npz_from_blob(b: bytes):
    buf = io.BytesIO(b)
    return np.load(buf, allow_pickle=False)

def open_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA mmap_size=268435456;")
    conn.commit()
    _ensure_schema(conn)
    _migrate_documents_table(conn)
    return conn

def upsert_document(conn: sqlite3.Connection, doc_id: str, full_text: str | None, *, uri: str | None = None):
    # Normalize types for SQLite
    if uri is not None and not isinstance(uri, str):
        uri = str(uri)
    if isinstance(full_text, bytes):
        try:
            full_text = full_text.decode("utf-8", errors="ignore")
        except Exception:
            full_text = None
    if full_text is not None and not isinstance(full_text, str):
        full_text = str(full_text)

    text_len = len(full_text) if isinstance(full_text, str) else None

    conn.execute(
        """
        INSERT INTO documents(doc_id, uri, full_text, text_length)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
          uri         = excluded.uri,
          full_text   = excluded.full_text,
          text_length = excluded.text_length
        """,
        (str(doc_id), uri, full_text, text_len)
    )

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

def upsert_doc_trie(cx: sqlite3.Connection, doc_id: str, trie_root: dict, idf: dict, chain_grams: dict):
    """
    trie_root: your serialized trie dict (nodes/edges or flat grams if you prefer)
    idf:      {gram: idf_value}
    chain_grams: {chain_id: {gram: count}}
    """
    _ensure_schema(cx)
    payload = _blob_from_json({"trie": trie_root, "idf": idf, "chain_grams": chain_grams})
    cx.execute("REPLACE INTO indices(doc_id, kind, payload) VALUES (?,?,?)",
               (doc_id, "trie", payload))
    cx.commit()

def upsert_doc_cooc(cx: sqlite3.Connection, doc_id: str, vocab: dict, rows, row_norms: np.ndarray):
    """
    vocab: {token: row_id}
    rows:  sparse CSR (PPMI or normalized); pass rows.data/indices/indptr/shape
    """
    _ensure_schema(cx)
    # store vocab as JSON
    cx.execute("REPLACE INTO indices(doc_id, kind, payload) VALUES (?,?,?)",
               (doc_id, "cooc_vocab", _blob_from_json({"vocab": vocab})))
    # store CSR as npz
    if sp is None or not sp.isspmatrix_csr(rows):
        raise RuntimeError("scipy.sparse.csr_matrix required for cooc rows")
    blob = _blob_from_npz(data=rows.data, indices=rows.indices, indptr=rows.indptr,
                          shape=np.array(rows.shape), norms=row_norms)
    cx.execute("REPLACE INTO indices(doc_id, kind, payload) VALUES (?,?,?)",
               (doc_id, "cooc_rows", blob))
    cx.commit()

def load_all_doc_tries(cx: sqlite3.Connection) -> dict:
    """Return {doc_id: {'trie':..., 'idf':..., 'chain_grams':...}}"""
    _ensure_schema(cx)
    out = {}
    for (doc_id, payload,) in cx.execute("SELECT doc_id, payload FROM indices WHERE kind='trie'"):
        out[doc_id] = _json_from_blob(payload)
    return out

def load_all_doc_coocs(cx: sqlite3.Connection) -> dict:
    """Return {doc_id: (vocab_dict, csr_rows, row_norms)}"""
    _ensure_schema(cx)
    vocabs = {}
    for (doc_id, payload,) in cx.execute("SELECT doc_id, payload FROM indices WHERE kind='cooc_vocab'"):
        vocabs[doc_id] = _json_from_blob(payload)["vocab"]
    rows = {}
    for (doc_id, payload,) in cx.execute("SELECT doc_id, payload FROM indices WHERE kind='cooc_rows'"):
        npz = _npz_from_blob(payload)
        data, indices, indptr = npz["data"], npz["indices"], npz["indptr"]
        shape = tuple(npz["shape"])
        norms = npz["norms"]
        csr = sp.csr_matrix((data, indices, indptr), shape=shape)
        rows[doc_id] = (csr, norms)
    out = {}
    for doc_id in set(vocabs) & set(rows):
        csr, norms = rows[doc_id]
        out[doc_id] = (vocabs[doc_id], csr, norms)
    return out

def count_indices(conn, kind: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM indices WHERE kind=?", (kind,))
    return cur.fetchone()[0]

def list_index_sizes(conn, kind: str, limit: int = 10):
    cur = conn.cursor()
    return cur.execute(
        "SELECT doc_id, LENGTH(payload) AS bytes "
        "FROM indices WHERE kind=? ORDER BY bytes DESC LIMIT ?",
        (kind, int(limit)),
    ).fetchall()

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
        # 1) documents
        upsert_document(conn, doc_id, production_output.get("full_text", ""), uri=uri)

        # 2) sentences
        sents = production_output.get("sentence_analyses") or []
        for s in sents:
            token_json = json.dumps(s.get("token_analysis") or {}, ensure_ascii=False)
            span_json  = json.dumps(s.get("span_analysis") or [], ensure_ascii=False)
            conn.execute(
                """
                INSERT OR REPLACE INTO sentences
                (doc_id, sentence_id, start, end, text, label, confidence, consensus, token_analysis_json, span_analysis_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(doc_id),
                    int(s.get("sentence_id", 0)),
                    int(s.get("doc_start", 0)),
                    int(s.get("doc_end", 0)),
                    s.get("sentence_text") or "",
                    s.get("classification", {}).get("label"),
                    s.get("classification", {}).get("confidence"),
                    s.get("classification", {}).get("consensus"),
                    token_json,
                    span_json,
                )
            )

        # 3) chains
        chains = (production_output.get("coreference_analysis") or {}).get("chains") or []
        for ch in chains:
            mentions_json = json.dumps(ch.get("mentions") or [], ensure_ascii=False)
            conn.execute(
                    """
                    INSERT OR REPLACE INTO chains(doc_id, chain_id, representative, mentions_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (str(doc_id), int(ch.get("chain_id", 0)), ch.get("representative") or "", mentions_json)
                    )

        conn.commit()

        upsert_clusters(conn, doc_id, production_output)
        if write_embeddings:
            sentences = [sa.get("sentence_text","") for sa in production_output.get("sentence_analyses",[])]
            upsert_sentence_embeddings(conn, doc_id, sentences)
    finally:
        conn.close()

def load_production_from_flexiconc(db_path: str, doc_id: str) -> Dict[str, Any]:
    conn = open_db(db_path)
    try:
        cur = conn.cursor()

        # --- Discover columns in documents
        doc_cols = {r[1] for r in cur.execute("PRAGMA table_info(documents)").fetchall()}
        has_full = ("full_text" in doc_cols)
        has_uri  = ("uri" in doc_cols)

        # --- Fetch document row
        if has_full and has_uri:
            row = cur.execute("SELECT uri, full_text FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
            uri, full_text = (row or (None, None))
        elif has_uri:
            row = cur.execute("SELECT uri FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
            uri = row[0] if row else None
            full_text = None
        else:
            full_text = None
            uri = None

        # --- Sentences (always present in your schema)
        srows = cur.execute("""
            SELECT sentence_id, start, end, text, label, confidence, consensus,
                   token_analysis_json, span_analysis_json
            FROM sentences
            WHERE doc_id=?
            ORDER BY sentence_id
        """, (doc_id,)).fetchall()

        sentence_analyses = []
        pieces = []
        for sid, st, en, txt, lab, conf, cons, tkj, spj in srows:
            # JSON fields can be NULL in legacy DBs
            tk = json.loads(tkj) if tkj else {}
            sp = json.loads(spj) if spj else []
            sentence_analyses.append({
                "sentence_id": sid,
                "sentence_text": txt or "",
                "doc_start": st, "doc_end": en,
                "classification": (
                    {"label": lab, "confidence": conf, "consensus": cons}
                    if lab is not None or conf is not None or cons is not None else {}
                ),
                "token_analysis": tk,
                "span_analysis": sp,
                "metadata": {}
            })
            pieces.append(txt or "")

        # --- Reconstruct full_text if missing
        if not full_text:
            full_text = " ".join(pieces).strip()

        # --- Chains (optional)
        chains = []
        try:
            crows = cur.execute("""
                SELECT chain_id, representative, mentions_json
                FROM chains
                WHERE doc_id=?
                ORDER BY chain_id
            """, (doc_id,)).fetchall()
            for cid, rep, mjson in crows:
                chains.append({
                    "chain_id": int(cid),
                    "representative": rep or "",
                    "mentions": (json.loads(mjson) if mjson else []),
                })
        except Exception:
            pass

        # --- Clusters (optional)
        cluster_analysis = None
        try:
            clrows = cur.execute("""
                SELECT cluster_id, members_json
                FROM clusters
                WHERE doc_id=?
                ORDER BY cluster_id
            """, (doc_id,)).fetchall()
            if clrows:
                clusters = [json.loads(members or "[]") for _, members in clrows]
                cluster_analysis = {
                    "clusters": clusters,
                    "clusters_dict": {str(i): c for i, c in enumerate(clusters)},
                    "graphs_json": {}
                }
        except Exception:
            cluster_analysis = None

        out = {
            "full_text": full_text or "",
            "document_uri": uri,
            "coreference_analysis": {"num_chains": len(chains), "chains": chains},
            "sentence_analyses": sentence_analyses,
        }
        if cluster_analysis is not None:
            out["cluster_analysis"] = cluster_analysis
        else:
            # omit key to keep payload small; your UI already (safely) handles missing clusters
            pass
        return out

    finally:
        conn.close()

"""
Created on Sat Aug 16 19:08:03 2025

@author: niran
"""

