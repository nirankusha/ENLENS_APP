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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import sqlite3, json, os, datetime
import contextlib
import gzip, io
import gzip, io, base64, re, math, argparse
import numpy as np
import logging

try:
    import scipy.sparse as sp
except Exception:
    sp = None  # allow import even if scipy not present


# If you already have an embedder in helper.py, we’ll use it; else lazy fallback/no-op.
try:
    from helper import get_sentence_embedding  # optional: (text) -> np.ndarray[float32]
except Exception:
    get_sentence_embedding = None

try:
    import faiss
except Exception:
    faiss = None

from ann_index import FaissIndex

# Cache for lightweight SentenceTransformer loaders (avoid repeated init).
_ST_MODELS: Dict[str, Any] = {}

logger = logging.getLogger(__name__)

GLOBAL_INDEX_DOC_ID = "__global__"

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

def _migrate_documents_table(conn: sqlite3.Connection, *, force_rebuild: bool = False):
    cur = conn.cursor()
    
    table_rows = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('documents', 'documents_old')"
    ).fetchall()
    tables = {row[0] for row in table_rows}

    
    try:
        info = cur.execute("PRAGMA table_info(documents)").fetchall()
    except sqlite3.OperationalError:
        info = []
   
    cols = {r[1] for r in info}

    doc_id_info = next((r for r in info if r[1] == "doc_id"), None)
    declared_type = ""
    is_integer_pk = False
    if doc_id_info:
        declared_type = (doc_id_info[2] or "").upper().lstrip()
        is_integer_pk = declared_type.startswith("INT") and bool(doc_id_info[5])
        
        
    needs_rebuild = bool(doc_id_info) and (force_rebuild or is_integer_pk)

    if needs_rebuild:
        cur.execute("DROP TABLE IF EXISTS documents_old")
        conn.commit()

        cur.execute("ALTER TABLE documents RENAME TO documents_old")
        conn.commit()

        _ensure_schema(conn)

        old_info = cur.execute("PRAGMA table_info(documents_old)").fetchall()
        old_cols = {r[1] for r in old_info}

        insert_cols = ["doc_id", "uri", "created_at", "text_length", "full_text"]
        select_exprs = [
            "CAST(doc_id AS TEXT) AS doc_id",
            "uri" if "uri" in old_cols else "NULL AS uri",
            "created_at" if "created_at" in old_cols else "NULL AS created_at",
            (
                "text_length"
                if "text_length" in old_cols
                else (
                    "CASE WHEN full_text IS NOT NULL THEN LENGTH(full_text) END AS text_length"
                    if "full_text" in old_cols
                    else "NULL AS text_length"
                )
            ),
            "full_text" if "full_text" in old_cols else "NULL AS full_text",
        ]

        cur.execute(
            f"INSERT INTO documents ({', '.join(insert_cols)}) "
            f"SELECT {', '.join(select_exprs)} FROM documents_old"
        )
        conn.commit()

        cur.execute("DROP TABLE IF EXISTS documents_old")
        conn.commit()

        info = cur.execute("PRAGMA table_info(documents)").fetchall()
        cols = {r[1] for r in info}
        
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

def _faiss_kind_for_model(model: str, aliases: Optional[Dict[str, str]] = None) -> str:
    if aliases and model in aliases:
        return aliases[model]
    base = model.split("/")[-1]
    base = re.sub(r"[^0-9a-zA-Z]+", "_", base).strip("_") or "model"
    return f"faiss_{base.lower()}"

def _default_nlist(count: int) -> int:
    if count <= 0:
        return 0
    # heuristic: sqrt-based, bounded for stability
    nlist = int(round(math.sqrt(count)))
    nlist = max(1, min(4096, nlist))
    return min(nlist, count)

def build_faiss_indices(
    conn: sqlite3.Connection,
    *,
    models: Optional[Iterable[str]] = None,
    aliases: Optional[Dict[str, str]] = None,
    doc_id: str = GLOBAL_INDEX_DOC_ID,
    nprobe: int = 16,
) -> Dict[str, Dict[str, Any]]:
    """Build FAISS ANN indices for sentence embeddings grouped by model.

    Returns a mapping ``{kind: {"model": str, "count": int, "dim": int}}`` for
    successfully persisted indices.
    """

    if faiss is None:
        raise RuntimeError("faiss is not installed. Install faiss-cpu to build indices.")

    _ensure_schema(conn)
    cur = conn.cursor()

    where_clause = ""
    params: Tuple[Any, ...] = tuple()
    if models:
        model_list = list(models)
        placeholders = ",".join("?" for _ in model_list)
        where_clause = f" WHERE model IN ({placeholders})"
        params = tuple(model_list)

    rows = cur.execute(
        "SELECT doc_id, sentence_id, model, dim, vector FROM embeddings" + where_clause,
        params,
    ).fetchall()

    by_model: Dict[str, List[Tuple[str, int, np.ndarray]]] = {}
    for doc_id_row, sentence_id, model_name, dim, blob in rows:
        if blob is None:
            continue
        vec = np.frombuffer(blob, dtype=np.float32)
        if dim and dim > 0 and vec.shape[0] != int(dim):
            vec = vec[: int(dim)]
        if vec.size == 0:
            continue
        vec = np.asarray(vec, dtype=np.float32)
        by_model.setdefault(model_name, []).append((doc_id_row, int(sentence_id), vec))

    summaries: Dict[str, Dict[str, Any]] = {}
    for model_name, entries in by_model.items():
        if not entries:
            continue
        target_dim = entries[0][2].shape[0]
        filtered = [(doc, sid, vec) for doc, sid, vec in entries if vec.shape[0] == target_dim]
        if not filtered:
            continue
        ids = [(doc, sid) for doc, sid, _ in filtered]
        matrix = np.vstack([vec for _, _, vec in filtered]).astype(np.float32)
        if matrix.ndim != 2:
            matrix = matrix.reshape(matrix.shape[0], -1)
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            continue
        # Normalize rows for cosine similarity.
        faiss.normalize_L2(matrix)

        nlist = _default_nlist(matrix.shape[0])
        if nlist <= 0:
            continue
        index = FaissIndex(matrix.shape[1], nlist=nlist, nprobe=nprobe)
        index.add(matrix)

        serialized = faiss.serialize_index(index.index)
        payload = {
            "model": model_name,
            "ids": ids,
            "count": len(ids),
            "dim": int(matrix.shape[1]),
            "nlist": int(index.nlist),
            "nprobe": int(index.nprobe),
            "index": base64.b64encode(serialized).decode("ascii"),
        }

        kind = _faiss_kind_for_model(model_name, aliases)
        conn.execute(
            "REPLACE INTO indices(doc_id, kind, payload) VALUES (?,?,?)",
            (doc_id, kind, _blob_from_json(payload)),
        )
        summaries[kind] = {"model": model_name, "count": len(ids), "dim": int(matrix.shape[1])}

    conn.commit()
    return summaries

def _decode_faiss_payload(payload_blob: bytes) -> Tuple[Any, List[Tuple[str, int]], Dict[str, Any]]:
    payload = _json_from_blob(payload_blob)
    if "index" not in payload:
        raise ValueError("Index payload missing serialized bytes")
    ids = [tuple(item) for item in payload.get("ids", [])]
    serialized = base64.b64decode(payload["index"])
    index = faiss.deserialize_index(serialized)
    if "nprobe" in payload and hasattr(index, "nprobe"):
        try:
            index.nprobe = int(payload["nprobe"])
        except Exception:
            pass
    return index, ids, payload

def query_concordance(
    conn: sqlite3.Connection,
    *,
    query_vector: Optional[np.ndarray] = None,
    query_text: Optional[str] = None,
    model: Optional[str] = None,
    kind: Optional[str] = None,
    aliases: Optional[Dict[str, str]] = None,
    doc_id: str = GLOBAL_INDEX_DOC_ID,
    topk: int = 10,
) -> List[Dict[str, Any]]:
    """Query a serialized FAISS index and return matching sentence rows."""

    if faiss is None:
        raise RuntimeError("faiss is not installed. Install faiss-cpu to query indices.")

    if kind is None:
        if not model:
            raise ValueError("Either 'kind' or 'model' must be provided.")
        kind = _faiss_kind_for_model(model, aliases)

    if query_vector is None:
        if query_text is None:
            raise ValueError("Provide either a query vector or query text.")
        if get_sentence_embedding is not None:
            query_vector = np.asarray(get_sentence_embedding(query_text), dtype=np.float32)
        else:
            from helper import sim_model

            query_vector = np.asarray(
                sim_model.encode([query_text])[0], dtype=np.float32
            )

    if query_vector is None:
        raise ValueError("Failed to obtain a query vector.")

    query_vector = np.asarray(query_vector, dtype=np.float32)
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    faiss.normalize_L2(query_vector)

    cur = conn.cursor()
    row = cur.execute(
        "SELECT payload FROM indices WHERE doc_id=? AND kind=?",
        (doc_id, kind),
    ).fetchone()
    if not row:
        raise KeyError(f"No FAISS index found for kind='{kind}'")

    index, ids, payload = _decode_faiss_payload(row[0])
    topk = max(1, int(topk))
    scores, indices = index.search(query_vector, topk)
    if scores.size == 0:
        return []

    hits: List[Tuple[str, int, float]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(ids):
            continue
        doc_sent = ids[idx]
        hits.append((doc_sent[0], int(doc_sent[1]), float(score)))

    if not hits:
        return []

    results: List[Dict[str, Any]] = []
    for doc_row, sent_id, score in hits:
        meta = cur.execute(
            """
            SELECT start, end, text, label, confidence, consensus
            FROM sentences
            WHERE doc_id=? AND sentence_id=?
            """,
            (doc_row, sent_id),
        ).fetchone()
        if not meta:
            continue
        start, end, text, label, confidence, consensus = meta
        results.append(
            {
                "doc_id": doc_row,
                "sentence_id": sent_id,
                "score": score,
                "start": start,
                "end": end,
                "text": text,
                "label": label,
                "confidence": confidence,
                "consensus": consensus,
                "model": payload.get("model"),
            }
        )

    return results

def _resolve_embedding_model_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    val = str(name).strip()
    if not val:
        return None
    lower = val.lower()
    mapping = {
        "mpnet": "sentence-transformers/paraphrase-mpnet-base-v2",
        "paraphrase-mpnet-base-v2": "sentence-transformers/paraphrase-mpnet-base-v2",
        "sentence-transformers/paraphrase-mpnet-base-v2": "sentence-transformers/paraphrase-mpnet-base-v2",
        "all-mpnet": "sentence-transformers/all-mpnet-base-v2",
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    }
    return mapping.get(lower, val)


def _get_sentence_transformer(model_name: str):
    if not model_name:
        return None
    if model_name in _ST_MODELS:
        return _ST_MODELS[model_name]
    if model_name.endswith("paraphrase-mpnet-base-v2"):
        try:
            from helper import sim_model
            _ST_MODELS[model_name] = sim_model
            return sim_model
        except Exception:
            pass
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        _ST_MODELS[model_name] = model
        return model
    except Exception as exc:
        logger.warning("SentenceTransformer(%s) unavailable: %s", model_name, exc)
        return None


def _encode_query_vector(text: str, model_name: str) -> Optional[np.ndarray]:
    model = _get_sentence_transformer(model_name)
    if model is None:
        return None
    try:
        vec = model.encode([text])[0]
    except Exception as exc:
        logger.warning("encode failed for model %s: %s", model_name, exc)
        return None
    try:
        return np.asarray(vec, dtype="float32")
    except Exception:
        return None


def _vector_concordance(
    conn: sqlite3.Connection,
    terms: List[str],
    limit: int,
    model_name: str,
    use_faiss: bool,
) -> Optional[Dict[str, Any]]:
    clean_terms = [t.strip() for t in terms if t and str(t).strip()]
    query_text = " ".join(clean_terms)
    if not query_text:
        return None

    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT e.doc_id, e.sentence_id, e.vector, e.dim,
                   s.text, s.start, s.end,
                   COALESCE(d.uri, s.doc_id) AS uri
            FROM embeddings e
            JOIN sentences s ON s.doc_id=e.doc_id AND s.sentence_id=e.sentence_id
            LEFT JOIN documents d ON d.doc_id=e.doc_id
            WHERE e.model=?
        """,
            (model_name,),
        )
        records = cur.fetchall()
    except Exception as exc:
        logger.warning("Failed to read embeddings for %s: %s", model_name, exc)
        return None

    vectors: List[np.ndarray] = []
    meta: List[Tuple[Any, Any, str, Any, Any, Any]] = []
    for doc_id, sid, blob, dim, text, start, end, uri in records:
        if blob is None or dim is None:
            continue
        try:
            vec = np.frombuffer(blob, dtype="float32")
        except Exception:
            continue
        if vec.size != int(dim):
            continue
        vectors.append(vec)
        meta.append((doc_id, sid, text or "", start, end, uri))

    if not vectors:
        return None

    X = np.vstack(vectors).astype("float32")
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms

    qvec = _encode_query_vector(query_text, model_name)
    if qvec is None:
        return None
    qn = qvec / (np.linalg.norm(qvec) + 1e-8)

    faiss_used = False
    order: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None

    if use_faiss and len(Xn) >= 8:
        try:
            from ann_index import FaissIndex

            n_samples = Xn.shape[0]
            nlist = max(1, min(256, int(np.sqrt(n_samples)) or 1))
            nprobe = min(32, max(4, int(np.sqrt(nlist)) or 4))
            index = FaissIndex(Xn.shape[1], nlist=nlist, nprobe=nprobe)
            index.add(Xn)
            D, I = index.search(qn.reshape(1, -1), topk=min(int(limit), n_samples))
            order = I[0]
            scores = D[0]
            faiss_used = True
        except Exception as exc:
            logger.info("FAISS search unavailable; falling back to brute-force: %s", exc)
            faiss_used = False

    if order is None:
        sims = Xn @ qn.reshape(-1, 1)
        sims = sims.reshape(-1)
        topk = min(int(limit), sims.shape[0])
        order = np.argsort(sims)[::-1][:topk]
        scores = sims[order]

    rows: List[Dict[str, Any]] = []
    emb_hits: List[np.ndarray] = []
    score_list: List[float] = []
    for rank, idx in enumerate(order):
        doc_id, sid, text, start, end, uri = meta[int(idx)]
        rows.append(
            {
                "doc_id": doc_id,
                "sentence_id": int(sid),
                "text": text,
                "start": start,
                "end": end,
                "path": uri or "",
                "score": float(scores[rank]) if scores is not None else None,
                "rank": rank + 1,
            }
        )
        emb_hits.append(X[int(idx)])
        score_list.append(float(scores[rank]) if scores is not None else 0.0)

    if not rows:
        return None

    return {
        "rows": rows,
        "embeddings": np.asarray(emb_hits, dtype="float32"),
        "meta": {
            "mode": "vector",
            "vector_backend": model_name,
            "faiss_used": faiss_used,
            "query_text": query_text,
            "total_candidates": len(meta),
            "scores": score_list,
        },
        "terms": clean_terms,
    }


def _lexical_concordance(
    conn: sqlite3.Connection,
    terms: List[str],
    mode: str,
    limit: int,
    model_name: Optional[str],
) -> Dict[str, Any]:
    clean_terms = [t.strip() for t in terms if t and str(t).strip()]
    if not clean_terms:
        return {
            "rows": [],
            "embeddings": None,
            "meta": {
                "mode": "lexical",
                "vector_backend": model_name,
                "faiss_used": False,
                "query_text": "",
                "total_candidates": 0,
                "scores": [],
            },
            "terms": [],
        }

    clauses = []
    params: List[Any] = []
    for t in clean_terms:
        clauses.append("LOWER(s.text) LIKE ?")
        params.append(f"%{t.lower()}%")
    joiner = " AND " if str(mode).upper() == "AND" else " OR "
    sql = (
        "SELECT s.doc_id, s.sentence_id, s.start, s.end, s.text, "
        "COALESCE(d.uri, s.doc_id) AS uri "
        "FROM sentences s "
        "LEFT JOIN documents d ON d.doc_id = s.doc_id "
        f"WHERE {joiner.join(clauses)} "
        "ORDER BY s.doc_id, s.sentence_id "
        "LIMIT ?"
    )
    params.append(int(limit))

    cur = conn.cursor()
    try:
        rows_raw = cur.execute(sql, params).fetchall()
    except Exception as exc:
        logger.warning("Lexical concordance failed: %s", exc)
        rows_raw = []

    rows: List[Dict[str, Any]] = []
    scores: List[float] = []
    for doc_id, sid, start, end, text, uri in rows_raw:
        txt = text or ""
        score = sum(txt.lower().count(t.lower()) for t in clean_terms)
        rows.append(
            {
                "doc_id": doc_id,
                "sentence_id": int(sid),
                "text": txt,
                "start": start,
                "end": end,
                "path": uri or "",
                "score": float(score),
            }
        )
        scores.append(float(score))

    return {
        "rows": rows,
        "embeddings": None,
        "meta": {
            "mode": "lexical",
            "vector_backend": model_name,
            "faiss_used": False,
            "query_text": " ".join(clean_terms),
            "total_candidates": len(rows_raw),
            "scores": scores,
        },
        "terms": clean_terms,
    }


def query_concordance(
    db_path: str,
    terms: List[str],
    mode: str = "AND",
    limit: int = 500,
    *,
    vector_backend: Optional[str] = None,
    use_faiss: bool = True,
) -> Dict[str, Any]:
    backend = _resolve_embedding_model_name(vector_backend)
    conn = open_db(db_path)
    try:
        if backend:
            try:
                vec_res = _vector_concordance(conn, terms, int(limit), backend, use_faiss)
            except Exception as exc:
                logger.warning("Vector concordance failed; falling back to lexical: %s", exc)
                vec_res = None
            if vec_res:
                return vec_res
        return _lexical_concordance(conn, terms, mode, int(limit), backend)
    finally:
        conn.close()

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

    
    sql = """
        INSERT INTO documents(doc_id, uri, full_text, text_length)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
          uri         = excluded.uri,
          full_text   = excluded.full_text,
          text_length = excluded.text_length
         """
    params = (str(doc_id), uri, full_text, text_len)

    try:
        conn.execute(sql, params)
    except sqlite3.IntegrityError as exc:
        if "datatype mismatch" not in str(exc).lower():
            raise
        _migrate_documents_table(conn, force_rebuild=True)
        conn.execute(sql, params)
               

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


def _call_encode(encode_fn: Callable[[Sequence[str]], Any],
                 sentences: Sequence[str]) -> List[Any]:
    """Call *encode_fn* with best-effort batching, falling back per sentence."""
    if not sentences:
        return []
    try:
        vectors = encode_fn(sentences)
    except TypeError:
        vectors = None
    except Exception as exc:  # propagate other errors to surface misconfiguration
        raise
    else:
        coerced = _coerce_vectors(vectors, len(sentences))
        if coerced is not None:
            return coerced
    # Fallback: try calling per sentence (handles encoders that expect str input)
    out: List[Any] = []
    for text in sentences:
        try:
            vec = encode_fn(text)  # type: ignore[arg-type]
        except TypeError:
            vec = encode_fn([text])  # type: ignore[arg-type]
        out.append(vec)
    return out


def _coerce_vectors(vectors: Any, expected_len: int) -> Optional[List[Any]]:
    if vectors is None:
        return None
    import numpy as np

    if isinstance(vectors, np.ndarray):
        if vectors.ndim == 1:
            if expected_len == 1:
                return [vectors]
            return None
        if vectors.ndim == 2:
            arr = [vectors[i] for i in range(min(expected_len, vectors.shape[0]))]
            return arr
    try:
        seq = list(vectors)
    except Exception:
        return None
    if len(seq) != expected_len:
        # allow callers that yield generators or mismatched lengths to fallback later
        return None
    return seq


def _resolve_encode_fn(encoder: Any) -> Optional[Callable[[Sequence[str]], Any]]:
    if encoder is None:
        return None
    if callable(encoder):
        return encoder
    encode_attr = getattr(encoder, "encode", None)
    if callable(encode_attr):
        return encode_attr
    return None


def upsert_sentence_embeddings(conn: sqlite3.Connection,
                               doc_id: str,
                               sentences: Sequence[str],
                               *,
                               model_name: str,
                               encode_fn: Callable[[Sequence[str]], Any] | None = None):
    import numpy as np

    if encode_fn is None:
        encode_fn = _resolve_encode_fn(get_sentence_embedding) if get_sentence_embedding else None
    if encode_fn is None:
        return
    vectors = _call_encode(encode_fn, list(sentences))
    rows = []
    for sid, vec in enumerate(vectors):
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            continue
        blob = _np_to_blob(arr)
        rows.append((doc_id, sid, model_name, int(arr.shape[-1]), blob))
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


def export_production_to_flexiconc(db_path: str,
                                   doc_id: str,
                                   production_output: Dict[str, Any],
                                   uri: Optional[str] = None,
                                   embedding_models: Optional[Dict[str, Any]] = None):
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
        if embedding_models:
            sentences = [sa.get("sentence_text", "") for sa in production_output.get("sentence_analyses", [])]
            for model_name, encoder in embedding_models.items():
                encode_fn = _resolve_encode_fn(encoder)
                try:
                    upsert_sentence_embeddings(conn, doc_id, sentences, model_name=model_name, encode_fn=encode_fn)
                except Exception:
                    # Surface issues but continue with other backends
                    import traceback

                    traceback.print_exc()
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

def _main_cli():
    parser = argparse.ArgumentParser(description="FlexiConc maintenance utilities")
    sub = parser.add_subparsers(dest="command")

    rebuild = sub.add_parser("build-faiss", help="Rebuild FAISS ANN indices from embeddings")
    rebuild.add_argument("db", help="Path to FlexiConc SQLite database")
    rebuild.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Restrict to one or more embedding model names",
    )
    rebuild.add_argument(
        "--alias",
        dest="aliases",
        action="append",
        default=None,
        help="Alias mapping in the form model=faiss_kind",
    )
    rebuild.add_argument("--nprobe", type=int, default=16, help="FAISS nprobe value")

    args = parser.parse_args()
    if args.command == "build-faiss":
        alias_map = {}
        if args.aliases:
            for pair in args.aliases:
                if "=" not in pair:
                    continue
                model_name, alias = pair.split("=", 1)
                alias_map[model_name] = alias
        conn = open_db(args.db)
        try:
            summary = build_faiss_indices(
                conn,
                models=args.models,
                aliases=(alias_map or None),
                nprobe=int(args.nprobe),
            )
        finally:
            conn.close()
        if not summary:
            print("No embeddings found to index.")
        else:
            for kind, meta in summary.items():
                print(
                    f"Stored {kind}: model={meta['model']} count={meta['count']} dim={meta['dim']}"
                )
    else:
        parser.print_help()


if __name__ == "__main__":
    _main_cli()

"""
Created on Sat Aug 16 19:08:03 2025

@author: niran
"""

