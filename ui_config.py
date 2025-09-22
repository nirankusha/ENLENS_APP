# -*- coding: utf-8 -*-
# ui_config.py
# Centralized config dicts used by the Streamlit app

INGEST_CONFIG = {
    "max_text_length": 100_000,
    "max_sentences": 50,
    "min_sentence_len": 25,
    "dedupe_near_duplicates": True,
    "keep_page_breaks": False,
    "emit_offsets": True,
}

SDG_CLASSIFIER_CONFIG = {
    "bert_checkpoint": "your/sdg-bert-checkpoint",
    "device": "auto",
    "max_seq_len": 512,
    "sim_checkpoint": "sentence-transformers/paraphrase-mpnet-base-v2",
    "sim_pooling": "mean",
    "agree_threshold": 0.10,
    "disagree_threshold": 0.25,
    "min_confidence": 0.40,
    "show_label_longform": True,
}

EXPLAIN_CONFIG = {
    "ig_enabled": True,
    "ig_steps": 32,
    "ig_smooth_grad": False,
    "ig_target": "predicted",

    "span_masking_enabled": True,
    "max_span_len": 4,
    "top_k_spans": 8,

    "kpe_top_k": 10,
    "kpe_threshold": 0.10,
    "kpe_checkpoint_path": "/content/BERT-KPE/checkpoints/bert2span.bin",

    "color_scale": "tomato_alpha",
    "min_abs_importance": 0.10,
    "topk_tokens_chips": 8,
    "topk_spans_chips": 6,
}


COREF_CONFIG = {
    "engine": "fastcoref",
    "device": "auto",
    "scope": "whole_document",
    "resolve_text": True,
    "window_sentences": 10,
    "window_stride": 5,
    "max_chains": 200,
    "max_mentions_per_chain": 50,
    "score_field": "score",
    "link_prune_threshold": 0.40,
    "attach_to_spans": True,
    "coref_shortlist_mode": "trie",   # "off" | "trie" | "cooc" | "both"
    "coref_shortlist_topk": 50,
    "coref_trie_tau": 0.18,
    "coref_cooc_tau": 0.18,
    "cooc_mode": "spacy",              # "spacy" | "hf"
    "cooc_hf_tokenizer": "",           
    "coref_use_pair_scorer": False,
    "coref_scorer_threshold": 0.25,
}

CORPUS_INDEX_CONFIG = {
    "sqlite_path": "flexiconc.sqlite",
    "index_limit": 20_000,
    "and_or_mode": "AND",
    "max_results_to_render": 200,
    "default_sentence_id": 0,
}

SCICO_CONFIG = {
    "use_shortlist": True,
    "faiss_topk": 32,
    "nprobe": 8,
    "add_lsh": True,
    "lsh_threshold": 0.80,
    "minhash_k": 5,
    "cheap_len_ratio": 0.25,
    "cheap_jaccard": 0.08,
    "use_coherence": False,
    "coherence_threshold": 0.55,
    "max_pairs": None,

    "clustering": "auto",
    "kmeans_k": 5,
    "community_on": "all",
    "community_method": "greedy",
    "prob_threshold": 0.55,
    "max_degree": 30,
    "top_edges_per_node": 30,

    "summarize": True,
    "summarize_on": "community",
    "summary_methods": ["centroid","xsum","presumm"],
    "xsum_sentences": 1,
    "sdg_topk": 3,
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "centroid_sim_threshold": 0.55,
    "centroid_top_n": 5,
    "centroid_store_vector": False
}

UI_CONFIG = {
    "mode": "document",
    "candidate_source": "span",
    "clickable_tokens": True,
    "auto_run_scico": False,
    "show_viz": False,
    "persist_to_flexiconc": False,
    "save_graph_meta": True,
    "debug": False,
    "persist_terms_across_sentences": False,
}


"""
Created on Mon Aug 25 15:02:13 2025

@author: niran
"""

