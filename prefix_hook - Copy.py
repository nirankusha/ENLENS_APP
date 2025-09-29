# -*- coding: utf-8 -*-
# prefix_hook.py
from typing import Optional, Dict, Callable, List
import torch
from token_trie import TokenTrie

def make_prefix_allowed_tokens_fn(
    tokenizer,
    trie: TokenTrie,
    *,
    trigger: Optional[Callable[[List[int], str], bool]] = None,
    hard: bool = True,
    allow_eos_on_terminal: bool = True,
):
    vocab_all = list(range(len(tokenizer)))
    state: Dict[int, Dict[str, int | bool]] = {}

    def fn(batch_id: int, input_ids: torch.Tensor):
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
Created on Mon Sep  8 11:43:36 2025

@author: niran
"""

