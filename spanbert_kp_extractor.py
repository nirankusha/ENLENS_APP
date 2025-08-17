"""
BERT-KPE Multi-word Keyphrase Extractor
======================================
A production-ready class for extracting multi-word keyphrases using BERT-KPE.
Corrected implementation with proper span extraction logic.

Usage:
    # Basic usage
    extractor = BertKpeExtractor(checkpoint_path="path/to/checkpoint")
    keyphrases = extractor.extract_keyphrases(text, top_k=10)
    
    # For debugging path issues
    extractor = BertKpeExtractor.__new__(BertKpeExtractor)
    extractor.checkpoint_path = "your/path"
    extractor.bert_kpe_repo_path = "your/repo/path"
    path_status = extractor.check_paths()  # Check before full initialization
"""

import os
import sys
import torch
import torch.nn as nn
import logging
import numpy as np
import json
import importlib.util
from transformers import BertTokenizer, BertConfig
import re
from typing import List, Tuple, Optional, Dict, Any

# Setup logging for the module
logging.basicConfig(level=logging.INFO)


class BertKpeExtractor:
    """
    Production-ready BERT-KPE keyphrase extractor with corrected multi-word span extraction.
    """
    
    @staticmethod
    def get_default_checkpoint_path(base_drive_path: str = "/content/drive/MyDrive") -> str:
        """
        Get the default checkpoint path based on the original BERT-KPE structure.
        
        Args:
            base_drive_path: Base path to your Google Drive or local directory
            
        Returns:
            Expected checkpoint path
        """
        return os.path.join(
            base_drive_path, 
            "ENLENS/BERT-KPE/checkpoints/bert2span/bert2span.openkp.bert.checkpoint"
        )
    
    @staticmethod
    def get_default_repo_path() -> str:
        """Get the default BERT-KPE repository path."""
        return "/content/BERT-KPE"
    
    def __init__(self, 
                 checkpoint_path: str,
                 bert_kpe_repo_path: str = '/content/BERT-KPE',
                 max_phrase_words: int = 5,
                 max_token: int = 512,
                 device: Optional[str] = None):
        """
        Initialize the BERT-KPE extractor.
        
        Args:
            checkpoint_path: Path to the BERT-KPE checkpoint file
            bert_kpe_repo_path: Path to the BERT-KPE repository
            max_phrase_words: Maximum number of words in extracted phrases
            max_token: Maximum number of tokens for input sequences
            device: Device to run inference on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.checkpoint_path = checkpoint_path
        self.bert_kpe_repo_path = bert_kpe_repo_path
        self.max_phrase_words = max_phrase_words
        self.max_token = max_token
        self.bert_output_dim = 768
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.config = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Log initialization info
        self.logger.info(f"Initializing BERT-KPE extractor...")
        self.logger.info(f"Checkpoint path: {self.checkpoint_path}")
        self.logger.info(f"Repository path: {self.bert_kpe_repo_path}")
        self.logger.info(f"Device: {self.device}")
        
        # Initialize the extractor
        self._setup_paths()
        self._load_model()
    
    def _setup_paths(self):
        """Setup Python paths for BERT-KPE repository."""
        if self.bert_kpe_repo_path not in sys.path:
            sys.path.insert(0, self.bert_kpe_repo_path)
            sys.path.insert(0, os.path.join(self.bert_kpe_repo_path, 'bertkpe'))
    
    def _load_native_bert2span_model(self):
        """Load the native BERT2Span model class from the repository."""
        try:
            bert2span_file = os.path.join(self.bert_kpe_repo_path, 'bertkpe/networks/Bert2Span.py')
            
            with open(bert2span_file, 'r') as f:
                bert2span_code = f.read()
            
            # Fix relative imports
            bert2span_code = bert2span_code.replace(
                'from ..transformers import BertPreTrainedModel, BertModel',
                'from transformers import BertPreTrainedModel, BertModel'
            )
            
            # Execute the modified code
            namespace = {}
            exec(bert2span_code, namespace)
            
            return namespace['BertForAttSpanExtractor']
            
        except Exception as e:
            self.logger.error(f"Error loading native model: {e}")
            raise
    
    def _load_configuration_and_tokenizer(self):
        """Load model configuration and tokenizer."""
        # The config.json and vocab.txt are in the same directory as the checkpoint
        base_path = os.path.dirname(self.checkpoint_path)
        config_path = os.path.join(base_path, 'config.json')
        vocab_path = os.path.join(base_path, 'vocab.txt')
        
        # Handle potential path variations (BERT-KPE vs BERT_KPE)
        if not os.path.exists(config_path):
            # Try alternative paths
            alt_base_path = base_path.replace('BERT-KPE', 'BERT_KPE').replace('BERT_KPE', 'BERT-KPE')
            alt_config_path = os.path.join(alt_base_path, 'config.json')
            alt_vocab_path = os.path.join(alt_base_path, 'vocab.txt')
            
            if os.path.exists(alt_config_path):
                config_path = alt_config_path
                vocab_path = alt_vocab_path
                base_path = alt_base_path
            else:
                # List available files for debugging
                self.logger.error(f"Config not found at: {config_path}")
                if os.path.exists(base_path):
                    available_files = os.listdir(base_path)
                    self.logger.error(f"Available files in {base_path}: {available_files}")
                else:
                    self.logger.error(f"Directory does not exist: {base_path}")
                    parent_dir = os.path.dirname(base_path)
                    if os.path.exists(parent_dir):
                        available_dirs = os.listdir(parent_dir)
                        self.logger.error(f"Available directories in {parent_dir}: {available_dirs}")
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = BertConfig(**config_dict)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise
        
        # Load tokenizer
        try:
            tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
            actual_vocab_size = len(tokenizer.vocab)
            
            # Fix vocab size mismatch if needed
            if config.vocab_size != actual_vocab_size:
                config.vocab_size = actual_vocab_size
                
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {e}")
            raise
        
        return config, tokenizer
    
    def _load_checkpoint(self):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_path
        
        if not os.path.exists(checkpoint_path):
            # Try alternative paths (handle BERT-KPE vs BERT_KPE variations)
            alternative_paths = [
                checkpoint_path.replace('BERT-KPE', 'BERT_KPE'),
                checkpoint_path.replace('BERT_KPE', 'BERT-KPE'),
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    checkpoint_path = alt_path
                    self.checkpoint_path = alt_path  # Update for consistency
                    break
            else:
                # List available files for debugging
                parent_dir = os.path.dirname(checkpoint_path)
                self.logger.error(f"Checkpoint not found at: {checkpoint_path}")
                if os.path.exists(parent_dir):
                    available_files = os.listdir(parent_dir)
                    self.logger.error(f"Available files in {parent_dir}: {available_files}")
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            import argparse
            torch.serialization.add_safe_globals([argparse.Namespace])
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.logger.info(f"Loaded checkpoint from: {checkpoint_path}")
            return checkpoint
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def _load_model(self):
        """Load the complete BERT2Span model."""
        try:
            # Load model class
            BertForAttSpanExtractor = self._load_native_bert2span_model()
            
            # Load config and tokenizer
            self.config, self.tokenizer = self._load_configuration_and_tokenizer()
            
            # Create model instance
            self.model = BertForAttSpanExtractor(self.config)
            
            # Load checkpoint
            checkpoint = self._load_checkpoint()
            
            if checkpoint is not None:
                state_dict = None
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                
                if state_dict:
                    # Clean parameter names
                    cleaned_state_dict = {}
                    for key, value in state_dict.items():
                        clean_key = key[7:] if key.startswith('module.') else key
                        cleaned_state_dict[clean_key] = value
                    
                    # Load weights
                    self.model.load_state_dict(cleaned_state_dict, strict=False)
            
            # Setup model for inference
            self.model.eval()
            self.model.to(self.device)
            
            self.logger.info(f"BERT-KPE model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _create_bert2span_inputs(self, text: str) -> Dict[str, Any]:
        """Create inputs exactly like bert2span_dataloader.py"""
        words = text.strip().split()
        doc_tokens = []
        valid_mask = []
        
        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                continue
            
            for i, token in enumerate(word_tokens):
                doc_tokens.append(token)
                valid_mask.append(1 if i == 0 else 0)
        
        # Add special tokens
        tokens = ['[CLS]'] + doc_tokens + ['[SEP]']
        valid_ids = [0] + valid_mask + [0]
        
        # Truncate if necessary
        if len(tokens) > self.max_token:
            tokens = tokens[:self.max_token-1] + ['[SEP]']
            valid_ids = valid_ids[:self.max_token-1] + [0]
        
        # Convert to IDs and create masks
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # Pad sequences
        while len(input_ids) < self.max_token:
            input_ids.append(0)
            attention_mask.append(0)
            valid_ids.append(0)
        
        # Create output tensors
        max_word_len = len(words)
        valid_output = torch.zeros(1, max_word_len, self.bert_output_dim)
        active_mask = torch.LongTensor(1, max_word_len).zero_()
        active_mask[0, :max_word_len].fill_(1)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'valid_ids': valid_ids,
            'valid_output': valid_output,
            'active_mask': active_mask,
            'words': words,
            'tokens': tokens
        }
    
    def _decode_span2phrase(self, orig_tokens: List[str], start_logit: List[float], 
                           end_logit: List[List[float]]) -> List[Tuple[str, float]]:
        """
        Core span extraction logic for multi-word keyphrases.
        
        Args:
            orig_tokens: Original word tokens
            start_logit: Start position probabilities
            end_logit: End position probabilities (triangular matrix)
            
        Returns:
            List of (phrase, score) tuples
        """
        seq_len = len(orig_tokens)
        phrase_list = []
        
        # Convert to probabilities if needed
        if isinstance(start_logit[0], (int, float)):
            start_probs = start_logit
        else:
            start_probs = torch.softmax(torch.tensor(start_logit), dim=0).tolist()
        
        if isinstance(end_logit[0], list):
            end_probs = end_logit
        else:
            end_probs = torch.softmax(torch.tensor(end_logit), dim=-1).tolist()
        
        # Extract spans with triangular constraint
        for start_idx in range(seq_len):
            start_prob = start_probs[start_idx]
            
            if start_prob < 0.05:
                continue
            
            best_spans = []
            
            for end_idx in range(start_idx, min(start_idx + self.max_phrase_words, seq_len)):
                if start_idx < len(end_probs) and end_idx < len(end_probs[start_idx]):
                    end_prob = end_probs[start_idx][end_idx]
                else:
                    continue
                
                span_score = start_prob * end_prob
                phrase_tokens = orig_tokens[start_idx:end_idx + 1]
                phrase = " ".join(phrase_tokens)
                
                # Quality filtering
                if (len(phrase.strip()) > 2 and
                    not all(word in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
                           for word in phrase_tokens)):
                    best_spans.append((phrase, span_score, start_idx, end_idx))
            
            # Select best span for this start position
            if best_spans:
                best_spans.sort(key=lambda x: x[1], reverse=True)
                best_phrase, best_score, _, _ = best_spans[0]
                
                if best_score > 0.1:
                    phrase_list.append((best_phrase, best_score))
        
        # Remove duplicates and sort
        seen_phrases = set()
        unique_phrases = []
        
        for phrase, score in phrase_list:
            phrase_norm = phrase.lower().strip()
            if phrase_norm not in seen_phrases:
                seen_phrases.add(phrase_norm)
                unique_phrases.append((phrase, score))
        
        unique_phrases.sort(key=lambda x: x[1], reverse=True)
        return unique_phrases
    
    def extract_keyphrases(self, text: str, top_k: int = 10, threshold: float = 0.05) -> List[Tuple[str, float]]:
        """
        Extract keyphrases from input text.
        
        Args:
            text: Input text to extract keyphrases from
            top_k: Maximum number of keyphrases to return
            threshold: Minimum confidence threshold for keyphrases
            
        Returns:
            List of (keyphrase, confidence_score) tuples, sorted by confidence
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not properly initialized")
        
        # Prepare inputs
        inputs = self._create_bert2span_inputs(text)
        
        # Convert to tensors
        input_ids = torch.tensor([inputs['input_ids']], device=self.device)
        attention_mask = torch.tensor([inputs['attention_mask']], device=self.device)
        valid_ids = torch.tensor([inputs['valid_ids']], device=self.device)
        valid_output = inputs['valid_output'].to(self.device)
        active_mask = inputs['active_mask'].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    valid_ids=valid_ids,
                    valid_output=valid_output,
                    active_mask=active_mask
                )
                
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    start_logits = outputs[0]
                    end_logits = outputs[1]
                    
                    # Process outputs
                    words = inputs['words']
                    lengths = [len(words)]
                    
                    s_logits = start_logits.data.cpu().tolist()
                    e_logits = end_logits.data.cpu()
                    
                    start_lists, end_lists = [], []
                    sum_len = 0
                    for l in lengths:
                        start_lists.append(s_logits[sum_len : (sum_len + l)])
                        end_lists.append(e_logits[sum_len : (sum_len + l), :l].tolist())
                        sum_len += l
                    
                    # Decode keyphrases
                    keyphrases = self._decode_span2phrase(
                        orig_tokens=words,
                        start_logit=start_lists[0],
                        end_logit=end_lists[0]
                    )
                    
                    # Filter and return results
                    filtered_keyphrases = [(phrase, score) for phrase, score in keyphrases if score > threshold]
                    return filtered_keyphrases[:top_k]
                
                else:
                    self.logger.warning("Unexpected model output format")
                    return []
                    
            except Exception as e:
                self.logger.error(f"Error in keyphrase extraction: {e}")
                return []
    
    def batch_extract_keyphrases(self, texts: List[str], top_k: int = 10, threshold: float = 0.05) -> List[List[Tuple[str, float]]]:
        """
        Extract keyphrases from multiple texts.
        
        Args:
            texts: List of input texts
            top_k: Maximum number of keyphrases per text
            threshold: Minimum confidence threshold
            
        Returns:
            List of keyphrase lists, one for each input text
        """
        results = []
        for text in texts:
            keyphrases = self.extract_keyphrases(text, top_k=top_k, threshold=threshold)
            results.append(keyphrases)
        return results
    
    def check_paths(self) -> Dict[str, bool]:
        """
        Check if all required paths exist. Useful for debugging.
        
        Returns:
            Dictionary with path existence status
        """
        base_path = os.path.dirname(self.checkpoint_path)
        paths_to_check = {
            'repository': self.bert_kpe_repo_path,
            'checkpoint': self.checkpoint_path,
            'checkpoint_dir': base_path,
            'config': os.path.join(base_path, 'config.json'),
            'vocab': os.path.join(base_path, 'vocab.txt'),
            'bert2span_file': os.path.join(self.bert_kpe_repo_path, 'bertkpe/networks/Bert2Span.py')
        }
        
        results = {}
        for name, path in paths_to_check.items():
            exists = os.path.exists(path)
            results[name] = exists
            if not exists:
                self.logger.warning(f"{name} not found: {path}")
                if os.path.isdir(os.path.dirname(path)):
                    available = os.listdir(os.path.dirname(path))
                    self.logger.info(f"Available in {os.path.dirname(path)}: {available}")
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "status": "loaded",
            "device": str(self.device),
            "total_parameters": total_params,
            "max_phrase_words": self.max_phrase_words,
            "max_tokens": self.max_token,
            "vocab_size": self.config.vocab_size if self.config else None
        }
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 20:53:44 2025

@author: niran
"""

