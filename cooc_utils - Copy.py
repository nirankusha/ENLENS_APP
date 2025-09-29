# -*- coding: utf-8 -*-
#cooc_utils.py
"""Utility helpers for configuring co-occurrence backends.
These helpers keep Streamlit apps lightweight by centralising
tokenizer loading / readiness checks so the UI can offer consistent
feedback without duplicating logic across entry points."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple


try:  # Prefer the tokenizer already initialised in helper.py
    from helper import bert_tokenizer as _DEFAULT_HF_TOKENIZER  # type: ignore
except Exception:  # pragma: no cover - helper assets missing in some envs
    _DEFAULT_HF_TOKENIZER = None


@lru_cache(maxsize=4)
def _load_hf_tokenizer(model_name: str):
    """Load and cache a HuggingFace tokenizer by name."""

    if not model_name:
        raise ValueError("model_name must be a non-empty string")
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("transformers.AutoTokenizer is unavailable") from exc

    return AutoTokenizer.from_pretrained(model_name)


def cooc_backend_ready(cfg: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Return (is_ready, message) for enabling shortlist modes using cooc."""

    mode = str(cfg.get("cooc_mode") or "spacy").lower()
    if mode != "hf":
        return True, None

    tok_name = str(cfg.get("cooc_hf_tokenizer") or "").strip()
    if tok_name:
        return True, None
    if _DEFAULT_HF_TOKENIZER is not None:
        return True, None

    return False, "Set cooc_hf_tokenizer to load a HuggingFace tokenizer or switch to the spaCy backend."


def resolve_cooc_backend(cfg: Dict[str, Any]) -> Tuple[str, Any, Optional[str]]:
    """Resolve (mode, tokenizer, warning) from config, with graceful fallback."""

    mode = str(cfg.get("cooc_mode") or "spacy").lower()
    tokenizer = None
    warn: Optional[str] = None

    if mode == "hf":
        tok_name = str(cfg.get("cooc_hf_tokenizer") or "").strip()
        if tok_name:
            try:
                tokenizer = _load_hf_tokenizer(tok_name)
            except Exception as exc:  # pragma: no cover - network / HF failures
                warn = f"Failed to load HuggingFace tokenizer '{tok_name}': {exc}"
                mode = "spacy"
        elif _DEFAULT_HF_TOKENIZER is not None:
            tokenizer = _DEFAULT_HF_TOKENIZER
        else:
            warn = "No HuggingFace tokenizer configured; falling back to spaCy co-occurrence."
            mode = "spacy"

    return mode, tokenizer, warn

"""
Created on Mon Sep 22 11:16:33 2025
@author: niran
"""

