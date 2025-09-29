# -*- coding: utf-8 -*-
# rag_trie_factory.py
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import hashlib

# Reuse the TokenTrie from earlier answer
class TokenTrie:
    __slots__ = ("_children", "_terminal", "_root")
    def __init__(self):
        self._children: List[Dict[int, int]] = [dict()]
        self._terminal: List[bool] = [False]
        self._root = 0
    def insert(self, ids: List[int]) -> None:
        node = self._root
        for t in ids:
            nxt = self._children[node].get(t)
            if nxt is None:
                nxt = len(self._children)
                self._children[node][t] = nxt
                self._children.append(dict())
                self._terminal.append(False)
            node = nxt
        self._terminal[node] = True
    def step(self, node: int, token_id: int) -> int:
        return self._children[node].get(token_id, -1)
    def allowed_from(self, node: int) -> List[int]:
        return list(self._children[node].keys())
    def is_terminal(self, node: int) -> bool:
        return self._terminal[node]
    @property
    def root(self) -> int: return self._root

def default_normalize(s: str) -> str:
    # minimal: strip trailing spaces; preserve inner/leading intentional spaces
    return s.rstrip()

def _tokenizer_fingerprint(tok) -> str:
    # Build a reasonably stable cache key for the tokenizer
    parts = [
        getattr(tok, "name_or_path", tok.__class__.__name__),
        str(getattr(tok, "vocab_size", "")),
        str(getattr(tok, "bos_token_id", "")),
        str(getattr(tok, "eos_token_id", "")),
        str(getattr(tok, "pad_token_id", "")),
        tok.__class__.__name__,
    ]
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()  # short, stable key

def _tokenize_flat(tok, text: str) -> List[int]:
    ids = tok(text, add_special_tokens=False).input_ids
    return ids[0] if isinstance(ids[0], list) else ids

@dataclass
class TrieFactory:
    phrases: List[str]
    normalize: Callable[[str], str] = default_normalize
    add_leading_space_variant: bool = True
    add_bos: bool = False
    add_eos: bool = False
    _cache: Dict[Tuple[str, bool, bool, bool], TokenTrie] = field(default_factory=dict)

    def _variants_for(self, tok, phrase: str) -> List[str]:
        """Generate robust variants across tokenizers."""
        base = self.normalize(phrase)
        variants = [base]
        if self.add_leading_space_variant:
            # Only add if it actually changes tokenization
            with_sp = " " + base
            if _tokenize_flat(tok, with_sp) != _tokenize_flat(tok, base):
                variants.append(with_sp)
        return variants

    def build_for(self, tok) -> TokenTrie:
        key = (_tokenizer_fingerprint(tok), self.add_leading_space_variant, self.add_bos, self.add_eos)
        if key in self._cache:
            return self._cache[key]

        trie = TokenTrie()
        bos = [tok.bos_token_id] if self.add_bos and tok.bos_token_id is not None else []
        eos = [tok.eos_token_id] if self.add_eos and tok.eos_token_id is not None else []

        for p in self.phrases:
            for v in self._variants_for(tok, p):
                ids = _tokenize_flat(tok, v)
                if not ids:
                    continue
                trie.insert(bos + ids + eos)

        self._cache[key] = trie
        return trie

# Convenience: build the prefix_allowed_tokens_fn
def make_prefix_allowed_tokens_fn(tokenizer, trie: TokenTrie,
                                  trigger: Optional[Callable[[List[int], str], bool]] = None,
                                  hard: bool = True,
                                  allow_eos_on_terminal: bool = True):
    vocab_all = list(range(len(tokenizer)))
    state: Dict[int, Dict[str, int | bool]] = {}
    def fn(batch_id: int, input_ids):
        st = state.get(batch_id)
        if st is None:
            st = {"node": trie.root, "active": trigger is None, "last_len": int(input_ids.shape[-1])}
            if trigger is not None:
                text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=True)
                st["active"] = bool(trigger(input_ids.tolist(), text))
            state[batch_id] = st

        prev_len, cur_len = st["last_len"], int(input_ids.shape[-1])
        st["last_len"] = cur_len

        if trigger is not None and not st["active"]:
            text = tokenizer.decode(input_ids.tolist(), skip_special_tokens=True)
            if trigger(input_ids.tolist(), text):
                st["active"] = True
                st["node"] = trie.root

        if not st["active"]:
            return vocab_all

        node = int(st["node"])
        for t in input_ids[prev_len:cur_len].tolist():
            nxt = trie.step(node, int(t))
            if nxt == -1:
                return [] if hard else vocab_all
            node = nxt
        st["node"] = node
        allowed = trie.allowed_from(node)
        if not allowed and trie.is_terminal(node) and allow_eos_on_terminal and tokenizer.eos_token_id is not None:
            return [tokenizer.eos_token_id]
        return allowed if allowed else ([] if hard else vocab_all)
    return fn

"""
Created on Mon Sep  8 11:44:48 2025

@author: niran
"""

