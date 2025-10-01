# -*- coding: utf-8 -*-
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from flexiconc_adapter import export_production_to_flexiconc


def _sample_production_output():
    return {
        "full_text": "Hello world.",
        "sentence_analyses": [
            {
                "sentence_id": 0,
                "doc_start": 0,
                "doc_end": 12,
                "sentence_text": "Hello world.",
                "classification": {"label": 1, "confidence": 0.5, "consensus": None},
                "token_analysis": {},
                "span_analysis": [],
            }
        ],
        "coreference_analysis": {"chains": []},
    }


def test_export_handles_legacy_integer_doc_id(tmp_path: Path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE documents(
                doc_id      \n                    INTEGER PRIMARY KEY,
                uri         TEXT,
                created_at  DATETIME,
                full_text   TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    # Should not raise when exporting even though doc_id column looked like INTEGER.
    export_production_to_flexiconc(str(db_path), "doc-1", _sample_production_output(), uri="example://doc")

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT doc_id, text_length FROM documents WHERE doc_id=?", ("doc-1",)).fetchone()
        assert row is not None
        assert row[0] == "doc-1"
        assert row[1] == len("Hello world.")
    finally:
        conn.close()
"""
Created on Wed Oct  1 11:43:24 2025

@author: niran
"""

