# -*- coding: utf-8 -*-
# token_trie.py
from typing import Dict, List

class TokenTrie:
    """Compact token-ID trie with O(1) step and cached node traversal."""
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
    def root(self) -> int:
        return self._root

"""
Created on Mon Sep  8 11:39:04 2025

@author: niran
"""

