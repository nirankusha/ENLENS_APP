# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Standalone Reverse Attribution System
====================================

A complete, self-contained system for attributing external summaries to source sentences.
No custom packages required - only standard libraries available on PyPI.

Installation:
    pip install sentence-transformers scikit-learn nltk

Usage:
    python standalone_reverse_attribution.py
"""

import numpy as np
import re
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

# Try importing optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence-transformers not found. Install with: pip install sentence-transformers")

try:
    import nltk
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("⚠️  nltk not found. Install with: pip install nltk")


@dataclass
class AttributionResult:
    """Result structure for attribution"""
    claim_id: int
    claim_text: str
    evidence: List[Dict]
    evidence_count: int


@dataclass  
class OverallResult:
    """Overall attribution results"""
    original_summary: str
    total_claims: int
    attributions: List[AttributionResult]
    stats: Dict


class StandaloneReverseAttributor:
    """
    Complete reverse attribution system with no external module dependencies.
    Finds source sentences that support claims in external summaries.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the attribution system
        
        Args:
            embedding_model: HuggingFace sentence transformer model name
        """
        self.embedding_model_name = embedding_model
        self.encoder = self._setup_encoder()
        self.corpus = {}  # doc_id -> sentences
        self.embeddings = {}  # doc_id -> sentence embeddings
        self.sentence_map = []  # Global mapping
        self._setup_nltk()
    
    def _setup_encoder(self):
        """Setup sentence encoder"""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required. Install with: "
                "pip install sentence-transformers"
            )
        
        print(f"Loading embedding model: {self.embedding_model_name}")
        return SentenceTransformer(self.embedding_model_name)
    
    def _setup_nltk(self):
        """Setup NLTK for sentence tokenization"""
        if HAS_NLTK:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
    
    def add_documents(self, documents: Dict[str, str]) -> None:
        """
        Add documents to the corpus for attribution
        
        Args:
            documents: Dictionary of {doc_id: document_text}
        """
        print(f"Processing {len(documents)} documents...")
        
        # Clear previous data
        self.corpus.clear()
        self.embeddings.clear()
        self.sentence_map.clear()
        
        for doc_id, text in documents.items():
            # Split into sentences
            sentences = self._split_sentences(text)
            if not sentences:
                continue
                
            self.corpus[doc_id] = sentences
            
            # Encode sentences
            print(f"  Encoding {len(sentences)} sentences from {doc_id}")
            embeddings = self.encoder.encode(sentences, show_progress_bar=False)
            self.embeddings[doc_id] = embeddings
            
            # Update global sentence mapping
            for i, _ in enumerate(sentences):
                self.sentence_map.append((doc_id, i))
        
        total_sentences = sum(len(sents) for sents in self.corpus.values())
        print(f"✅ Loaded {total_sentences} sentences from {len(documents)} documents")
    
    def attribute_summary(self, 
                         summary: str, 
                         top_k: int = 3,
                         min_similarity: float = 0.3,
                         claim_extraction_method: str = "sentences") -> OverallResult:
        """
        Find source sentences that support claims in the summary
        
        Args:
            summary: External summary text to attribute
            top_k: Number of supporting sentences per claim
            min_similarity: Minimum cosine similarity threshold (0-1)
            claim_extraction_method: "sentences" or "semantic_chunks"
            
        Returns:
            Complete attribution results
        """
        if not self.corpus:
            raise ValueError("No documents loaded. Call add_documents() first.")
        
        # Extract claims from summary
        if claim_extraction_method == "sentences":
            claims = self._split_sentences(summary)
        elif claim_extraction_method == "semantic_chunks":
            claims = self._extract_semantic_chunks(summary)
        else:
            claims = [summary]  # Treat entire summary as one claim
        
        if not claims:
            claims = [summary]  # Fallback
        
        print(f"Extracted {len(claims)} claims from summary")
        
        # Find attributions for each claim
        attributions = []
        for i, claim in enumerate(claims):
            print(f"  Processing claim {i+1}/{len(claims)}")
            supporting_evidence = self._find_supporting_sentences(
                claim, top_k, min_similarity
            )
            
            attributions.append(AttributionResult(
                claim_id=i + 1,
                claim_text=claim,
                evidence=supporting_evidence,
                evidence_count=len(supporting_evidence)
            ))
        
        # Compute statistics
        stats = self._compute_stats(attributions)
        
        return OverallResult(
            original_summary=summary,
            total_claims=len(claims),
            attributions=attributions,
            stats=stats
        )
    
    def _find_supporting_sentences(self, 
                                  claim: str, 
                                  top_k: int, 
                                  min_similarity: float) -> List[Dict]:
        """Find top-k most similar sentences to a claim"""
        
        # Encode the claim
        claim_embedding = self.encoder.encode([claim], show_progress_bar=False)
        
        # Collect all sentence embeddings with metadata
        all_embeddings = []
        sentence_metadata = []
        
        for doc_id, doc_embeddings in self.embeddings.items():
            for i, embedding in enumerate(doc_embeddings):
                all_embeddings.append(embedding)
                sentence_metadata.append({
                    'doc_id': doc_id,
                    'sentence_idx': i,
                    'sentence_text': self.corpus[doc_id][i]
                })
        
        if not all_embeddings:
            return []
        
        # Compute similarities
        all_embeddings = np.array(all_embeddings)
        similarities = cosine_similarity(claim_embedding, all_embeddings)[0]
        
        # Get top-k most similar above threshold
        top_indices = np.argsort(similarities)[::-1]
        
        supporting_evidence = []
        for idx in top_indices:
            if len(supporting_evidence) >= top_k:
                break
                
            similarity = similarities[idx]
            if similarity >= min_similarity:
                metadata = sentence_metadata[idx]
                supporting_evidence.append({
                    'source_doc': metadata['doc_id'],
                    'sentence': metadata['sentence_text'],
                    'similarity': float(similarity),
                    'strength': self._similarity_to_strength(similarity),
                    'sentence_idx': metadata['sentence_idx']
                })
        
        return supporting_evidence
    
    def _extract_semantic_chunks(self, text: str) -> List[str]:
        """Extract semantic chunks instead of individual sentences"""
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 2:
            return sentences
        
        # Group sentences by semantic similarity
        sentence_embeddings = self.encoder.encode(sentences, show_progress_bar=False)
        
        # Simple clustering approach
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            similarity = cosine_similarity(
                [sentence_embeddings[i-1]], 
                [sentence_embeddings[i]]
            )[0][0]
            
            if similarity > 0.7:  # High similarity threshold
                current_chunk.append(sentences[i])
            else:
                # Start new chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK or regex fallback"""
        
        if HAS_NLTK:
            try:
                sentences = nltk.sent_tokenize(text)
                return [s.strip() for s in sentences if len(s.strip()) > 10]
            except:
                pass  # Fall back to regex
        
        # Regex fallback
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _similarity_to_strength(self, similarity: float) -> str:
        """Convert similarity score to human-readable strength"""
        if similarity >= 0.8:
            return "Very Strong"
        elif similarity >= 0.6:
            return "Strong"
        elif similarity >= 0.4:
            return "Moderate"
        else:
            return "Weak"
    
    def _compute_stats(self, attributions: List[AttributionResult]) -> Dict:
        """Compute attribution statistics"""
        total_claims = len(attributions)
        attributed_claims = sum(1 for attr in attributions if attr.evidence)
        
        all_similarities = []
        source_docs = set()
        
        for attr in attributions:
            for evidence in attr.evidence:
                all_similarities.append(evidence['similarity'])
                source_docs.add(evidence['source_doc'])
        
        return {
            'coverage': attributed_claims / total_claims if total_claims > 0 else 0,
            'avg_similarity': np.mean(all_similarities) if all_similarities else 0,
            'sources_used': len(source_docs),
            'total_evidence': len(all_similarities),
            'strong_attributions': sum(1 for s in all_similarities if s >= 0.6),
            'weak_attributions': sum(1 for s in all_similarities if s < 0.4)
        }


class SimpleSummarizer:
    """Simple summarization methods that don't require external APIs"""
    
    @staticmethod
    def extractive_summary(text: str, num_sentences: int = 3) -> str:
        """Simple extractive summarization using TF-IDF"""
        if not HAS_SENTENCE_TRANSFORMERS:
            # Fallback to first N sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            return '. '.join(sentences[:num_sentences]) + '.'
        
        # Use sentence embeddings for better summarization
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= num_sentences:
            return text
        
        # Encode sentences
        embeddings = encoder.encode(sentences)
        
        # Compute sentence similarities to document centroid
        doc_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
        similarities = cosine_similarity(embeddings, doc_embedding).flatten()
        
        # Select top sentences
        top_indices = np.argsort(similarities)[-num_sentences:]
        top_indices = sorted(top_indices)  # Maintain order
        
        summary_sentences = [sentences[i] for i in top_indices]
        return '. '.join(summary_sentences) + '.'


def print_attribution_results(result: OverallResult) -> None:
    """Pretty print attribution results"""
    print("\n" + "="*60)
    print("🔍 REVERSE ATTRIBUTION RESULTS")
    print("="*60)
    
    print(f"\n📝 Original Summary:")
    print(f"   {result.original_summary}")
    
    print(f"\n📊 Statistics:")
    print(f"   Claims Found: {result.total_claims}")
    print(f"   Coverage: {result.stats['coverage']:.1%}")
    print(f"   Average Similarity: {result.stats['avg_similarity']:.3f}")
    print(f"   Sources Used: {result.stats['sources_used']}")
    print(f"   Strong Attributions: {result.stats['strong_attributions']}")
    
    print(f"\n" + "-"*60)
    print("📋 DETAILED ATTRIBUTIONS")
    print("-"*60)
    
    for attr in result.attributions:
        print(f"\n💡 CLAIM {attr.claim_id}: {attr.claim_text}")
        
        if attr.evidence:
            print(f"   Supporting Evidence ({len(attr.evidence)} sources):")
            for i, evidence in enumerate(attr.evidence, 1):
                print(f"   [{i}] {evidence['sentence']}")
                print(f"       📂 Source: {evidence['source_doc']} | "
                      f"🔗 Similarity: {evidence['similarity']:.3f} | "
                      f"💪 Strength: {evidence['strength']}")
        else:
            print("   ❌ No supporting evidence found above threshold")


def demo_basic_usage():
    """Demonstrate basic usage"""
    print("🚀 DEMO: Basic Reverse Attribution")
    print("-" * 50)
    
    # Sample documents (replace with your corpus)
    documents = {
        "climate_science": """
        Global warming has accelerated due to increased greenhouse gas emissions from fossil fuels.
        Rising temperatures are causing more frequent extreme weather events including hurricanes and droughts.
        Sea levels are rising due to thermal expansion and melting ice caps.
        """,
        
        "economic_impact": """
        Climate change poses significant risks to the global economy.
        Agricultural productivity is declining in many regions due to changing weather patterns.
        Infrastructure damage from extreme weather is costing billions annually.
        Renewable energy investments are growing as costs decrease rapidly.
        """,
        
        "policy_response": """
        International climate agreements like the Paris Accord aim to limit global warming.
        Carbon pricing mechanisms are being implemented in many countries.
        Governments are setting renewable energy targets and phasing out fossil fuel subsidies.
        """
    }
    
    # External summary to attribute (from any source)
    external_summary = """
    Climate change is primarily driven by fossil fuel emissions, leading to rising temperatures and extreme weather.
    This creates significant economic challenges including agricultural losses and infrastructure damage.
    International agreements and carbon pricing policies are being implemented to address these issues.
    """
    
    try:
        # Setup attribution system
        attributor = StandaloneReverseAttributor()
        attributor.add_documents(documents)
        
        # Perform attribution
        result = attributor.attribute_summary(
            summary=external_summary,
            top_k=2,
            min_similarity=0.25
        )
        
        # Display results
        print_attribution_results(result)
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return None
    
    return result


def demo_with_generated_summary():
    """Demo using generated summary"""
    print("\n\n🤖 DEMO: Generated Summary Attribution")
    print("-" * 50)
    
    documents = {
        "ai_research": "Artificial intelligence has made significant breakthroughs in natural language processing and computer vision. Large language models can now generate human-like text.",
        "ai_applications": "AI is being deployed in healthcare for medical diagnosis and in finance for fraud detection. Autonomous vehicles use AI for navigation and safety.",
        "ai_ethics": "AI development raises concerns about bias, privacy, and job displacement. Researchers are working on making AI systems more transparent and fair."
    }
    
    try:
        # Generate summary using simple extractive method
        combined_text = " ".join(documents.values())
        generated_summary = SimpleSummarizer.extractive_summary(combined_text, num_sentences=2)
        
        print(f"📄 Generated Summary: {generated_summary}")
        
        # Attribute the generated summary
        attributor = StandaloneReverseAttributor()
        attributor.add_documents(documents)
        
        result = attributor.attribute_summary(
            summary=generated_summary,
            top_k=1,
            min_similarity=0.2
        )
        
        print_attribution_results(result)
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")


def main():
    """Run demos"""
    print("🎯 STANDALONE REVERSE ATTRIBUTION SYSTEM")
    print("=" * 60)
    
    # Check dependencies
    if not HAS_SENTENCE_TRANSFORMERS:
        print("❌ sentence-transformers is required but not installed")
        print("💡 Install with: pip install sentence-transformers")
        return
    
    # Run demonstrations
    demo_basic_usage()
    demo_with_generated_summary()
    
    print(f"\n✅ Reverse attribution completed successfully!")
    print(f"💡 This system works with any external summary from any source.")


if __name__ == "__main__":
    main()
"""
Created on Sat Sep 27 21:13:39 2025

@author: niran
"""

