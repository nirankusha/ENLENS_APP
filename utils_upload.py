# -*- coding: utf-8 -*-
# utils_upload.py
from pathlib import Path
from tempfile import NamedTemporaryFile
import os

def persist_upload(uploaded_file, *, base_dir="/content/uploads") -> str:
    """
    Save a Streamlit UploadedFile to a stable path and return it.
    Safe across reruns; also rehydrates if the temp file disappears.
    """
    os.makedirs(base_dir, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with NamedTemporaryFile(delete=False, dir=base_dir, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

from pathlib import Path
import os, uuid

def save_uploaded_pdf(file) -> str:
    """Save a Streamlit UploadedFile to /tmp and return the exact path."""
    suffix = ".pdf"
    name = f"uploaded_{uuid.uuid4().hex}{suffix}"
    out = Path("/tmp") / name
    with open(out, "wb") as f:
        f.write(file.getbuffer())
    return str(out)


def ensure_file(path: str, *, bytes_fallback: bytes | None = None) -> str:
    """
    If the saved file was GC'd between reruns, recreate it from bytes.
    """
    p = Path(path)
    if not p.exists() and bytes_fallback:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bytes_fallback)
    return str(p)

"""
Created on Mon Aug 25 16:12:35 2025

@author: niran
"""

