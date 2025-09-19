# -*- coding: utf-8 -*-
# trie_factory_spacy.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable, Optional
import hashlib
from token_trie import TokenTrie

def _hf_tokenizer_fingerprint(tok) -> str:
    parts = [
        getattr(tok, "name_or_path", tok.__class__.__name__),
        str(getattr(tok, "vocab_size", "")),
        str(getattr(tok, "bos_token_id", "")),
        str(getattr(tok, "eos_token_id", "")),
        str(getattr(tok, "pad_token_id", "")),
        tok.__class__.__name__,
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

@dataclass
class SpacyTrieFactory:
    nlp: "spacy.language.Language"
    phrases: List[str]
    # normalization options
    lowercase: bool = False
    use_lemma: bool = False           # use token.lemma_ instead of token.text
    keep_punct: bool = True           # keep punctuation tokens
    strip_spaces: bool = True         # strip each token text/lemma
    # tokenizer-compat options
    add_leading_space_variant: bool = True
    add_bos: bool = False
    add_eos: bool = False
    # cache
    _cache: Dict[Tuple[str, bool, bool, bool], TokenTrie] = field(default_factory=dict)

    def _spacy_tokens(self, phrase: str) -> List[str]:
        doc = self.nlp(phrase)
        toks: List[str] = []
        for t in doc:
            if not self.keep_punct and t.is_punct:
                continue
            s = (t.lemma_ if self.use_lemma else t.text)
            if self.strip_spaces:
                s = s.strip()
            if self.lowercase:
                s = s.lower()
            if s != "":
                toks.append(s)
        return toks

    def _variants_for(self, tok, word_tokens: List[str]) -> List[List[str]]:
        """
        Produce variants robust to space-sensitive tokenizers.
        Variant A: original word tokens.
        Variant B: first token prefixed with a space (helps Llama/Gemma/BPE models).
        Only add B when it changes tokenization.
        """
        variants = [word_tokens]
        if not word_tokens or not self.add_leading_space_variant:
            return variants

        first = word_tokens[0]
        spaced = " " + first
        # Check if prefixing a space changes HF tokenization
        ids_a = tok(word_tokens, is_split_into_words=True, add_special_tokens=False).input_ids
        ids_b = tok([spaced] + word_tokens[1:], is_split_into_words=True, add_special_tokens=False).input_ids
        if ids_a != ids_b:
            variants.append([spaced] + word_tokens[1:])
        return variants

    def build_for(self, hf_tok) -> TokenTrie:
        key = (_hf_tokenizer_fingerprint(hf_tok), self.add_leading_space_variant, self.add_bos, self.add_eos)
        if key in self._cache:
            return self._cache[key]

        trie = TokenTrie()
        for phr in self.phrases:
            words = self._spacy_tokens(phr)
            if not words:
                continue
            for wseq in self._variants_for(hf_tok, words):
                # Convert spaCy words -> HF token IDs (respect word boundaries)
                enc = hf_tok(wseq, is_split_into_words=True, add_special_tokens=False)
                # Some tokenizers return nested lists for batch; make sure it’s flat
                ids = enc.input_ids
                if isinstance(ids[0], list):
                    ids = ids[0]
                if self.add_bos and hf_tok.bos_token_id is not None:
                    ids = [hf_tok.bos_token_id] + ids
                if self.add_eos and hf_tok.eos_token_id is not None:
                    ids = ids + [hf_tok.eos_token_id]
                if ids:
                    trie.insert(ids)

        self._cache[key] = trie
        return trie

"""
Created on Mon Sep  8 11:43:00 2025

@author: niran
"""

