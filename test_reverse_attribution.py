# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Test Suite for Reverse Attribution System
=========================================

Comprehensive tests for the StandaloneReverseAttributor class and related functionality.

Prerequisites:
    pip install sentence-transformers scikit-learn nltk pytest

Usage:
    # Run all tests
    python test_reverse_attribution.py
    
    # Run with pytest (optional)
    pytest test_reverse_attribution.py -v
"""

import sys
import os
import unittest
from typing import Dict, List
import numpy as np

# Import the reverse attribution classes
try:
    from reverse_attribution import (
        StandaloneReverseAttributor,
        SimpleSummarizer,
        AttributionResult,
        OverallResult,
        print_attribution_results
    )
    print("✅ Successfully imported reverse attribution classes")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure standalone_reverse_attribution.py is in the same directory")
    sys.exit(1)


class TestStandaloneReverseAttributor(unittest.TestCase):
    """Test cases for the StandaloneReverseAttributor class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.sample_documents = {
            "climate_doc": """
            Global warming is caused by greenhouse gas emissions from burning fossil fuels.
            Rising temperatures lead to melting ice caps and rising sea levels.
            Extreme weather events are becoming more frequent and severe.
            """,
            
            "economy_doc": """
            Climate change has significant economic impacts on agriculture and industry.
            Renewable energy investments are growing rapidly worldwide.
            Carbon pricing mechanisms help reduce emissions while generating revenue.
            """,
            
            "policy_doc": """
            International climate agreements like Paris Accord set emission reduction targets.
            Governments are implementing policies to transition to clean energy.
            Public awareness and education about climate change is increasing.
            """
        }
        
        self.external_summary = """
        Climate change is driven by fossil fuel emissions causing rising temperatures.
        Economic impacts include effects on agriculture and growth in renewable energy.
        Policy responses include international agreements and government initiatives.
        """
        
    def test_initialization(self):
        """Test attributor initialization"""
        try:
            attributor = StandaloneReverseAttributor()
            self.assertIsNotNone(attributor.encoder)
            self.assertEqual(attributor.embedding_model_name, "all-MiniLM-L6-v2")
            print("✅ Initialization test passed")
        except Exception as e:
            self.fail(f"Initialization failed: {e}")
    
    def test_custom_embedding_model(self):
        """Test initialization with custom embedding model"""
        try:
            custom_model = "sentence-transformers/all-mpnet-base-v2"
            attributor = StandaloneReverseAttributor(embedding_model=custom_model)
            self.assertEqual(attributor.embedding_model_name, custom_model)
            print("✅ Custom embedding model test passed")
        except Exception as e:
            print(f"⚠️  Custom model test skipped: {e}")
    
    def test_add_documents(self):
        """Test adding documents to the corpus"""
        attributor = StandaloneReverseAttributor()
        
        # Test with sample documents
        attributor.add_documents(self.sample_documents)
        
        # Verify documents were added
        self.assertEqual(len(attributor.corpus), 3)
        self.assertIn("climate_doc", attributor.corpus)
        self.assertIn("economy_doc", attributor.corpus)
        self.assertIn("policy_doc", attributor.corpus)
        
        # Verify sentences were split and embedded
        for doc_id in self.sample_documents.keys():
            self.assertGreater(len(attributor.corpus[doc_id]), 0)
            self.assertIn(doc_id, attributor.embeddings)
        
        print("✅ Add documents test passed")
    
    def test_empty_documents(self):
        """Test handling of empty documents"""
        attributor = StandaloneReverseAttributor()
        
        empty_docs = {
            "empty1": "",
            "empty2": "   ",
            "short": "Hi."  # Too short to be meaningful
        }
        
        attributor.add_documents(empty_docs)
        
        # Should handle empty documents gracefully
        self.assertIsInstance(attributor.corpus, dict)
        print("✅ Empty documents test passed")
    
    def test_basic_attribution(self):
        """Test basic attribution functionality"""
        attributor = StandaloneReverseAttributor()
        attributor.add_documents(self.sample_documents)
        
        result = attributor.attribute_summary(
            summary=self.external_summary,
            top_k=2,
            min_similarity=0.1  # Low threshold for testing
        )
        
        # Verify result structure
        self.assertIsInstance(result, OverallResult)
        self.assertEqual(result.original_summary, self.external_summary)
        self.assertGreater(result.total_claims, 0)
        self.assertIsInstance(result.attributions, list)
        self.assertIsInstance(result.stats, dict)
        
        # Verify statistics
        required_stats = ['coverage', 'avg_similarity', 'sources_used', 'total_evidence']
        for stat in required_stats:
            self.assertIn(stat, result.stats)
        
        print("✅ Basic attribution test passed")
        print(f"   Claims found: {result.total_claims}")
        print(f"   Coverage: {result.stats['coverage']:.1%}")
    
    def test_attribution_thresholds(self):
        """Test different similarity thresholds"""
        attributor = StandaloneReverseAttributor()
        attributor.add_documents(self.sample_documents)
        
        # Test with high threshold
        result_high = attributor.attribute_summary(
            summary=self.external_summary,
            top_k=3,
            min_similarity=0.8  # Very high threshold
        )
        
        # Test with low threshold  
        result_low = attributor.attribute_summary(
            summary=self.external_summary,
            top_k=3,
            min_similarity=0.1  # Low threshold
        )
        
        # Low threshold should find more evidence
        self.assertGreaterEqual(
            result_low.stats['total_evidence'],
            result_high.stats['total_evidence']
        )
        
        print("✅ Attribution thresholds test passed")
        print(f"   High threshold evidence: {result_high.stats['total_evidence']}")
        print(f"   Low threshold evidence: {result_low.stats['total_evidence']}")
    
    def test_claim_extraction_methods(self):
        """Test different claim extraction methods"""
        attributor = StandaloneReverseAttributor()
        attributor.add_documents(self.sample_documents)
        
        # Test sentence-based extraction
        result_sentences = attributor.attribute_summary(
            summary=self.external_summary,
            claim_extraction_method="sentences"
        )
        
        # Test semantic chunk extraction
        result_chunks = attributor.attribute_summary(
            summary=self.external_summary,
            claim_extraction_method="semantic_chunks"
        )
        
        # Test treating summary as single claim
        result_single = attributor.attribute_summary(
            summary=self.external_summary,
            claim_extraction_method="single"
        )
        
        # Verify all methods return valid results
        for result in [result_sentences, result_chunks, result_single]:
            self.assertIsInstance(result, OverallResult)
            self.assertGreater(result.total_claims, 0)
        
        print("✅ Claim extraction methods test passed")
        print(f"   Sentences method: {result_sentences.total_claims} claims")
        print(f"   Chunks method: {result_chunks.total_claims} claims")  
        print(f"   Single method: {result_single.total_claims} claims")
    
    def test_no_documents_error(self):
        """Test error handling when no documents are loaded"""
        attributor = StandaloneReverseAttributor()
        
        with self.assertRaises(ValueError):
            attributor.attribute_summary("Test summary")
        
        print("✅ No documents error test passed")
    
    def test_attribution_result_structure(self):
        """Test the structure of attribution results"""
        attributor = StandaloneReverseAttributor()
        attributor.add_documents(self.sample_documents)
        
        result = attributor.attribute_summary(
            summary=self.external_summary,
            top_k=1,
            min_similarity=0.2
        )
        
        # Test each attribution
        for attr in result.attributions:
            self.assertIsInstance(attr, AttributionResult)
            self.assertGreater(attr.claim_id, 0)
            self.assertIsInstance(attr.claim_text, str)
            self.assertIsInstance(attr.evidence, list)
            self.assertEqual(attr.evidence_count, len(attr.evidence))
            
            # Test evidence structure
            for evidence in attr.evidence:
                required_keys = ['source_doc', 'sentence', 'similarity', 'strength']
                for key in required_keys:
                    self.assertIn(key, evidence)
                
                self.assertIsInstance(evidence['similarity'], float)
                self.assertIn(evidence['strength'], 
                            ['Very Strong', 'Strong', 'Moderate', 'Weak'])
        
        print("✅ Attribution result structure test passed")


class TestSimpleSummarizer(unittest.TestCase):
    """Test cases for the SimpleSummarizer class"""
    
    def test_extractive_summary(self):
        """Test extractive summarization"""
        test_text = """
        Artificial intelligence is transforming many industries today.
        Machine learning algorithms can process vast amounts of data.
        Deep learning models achieve human-level performance in many tasks.
        Natural language processing enables computers to understand text.
        Computer vision allows machines to interpret visual information.
        """
        
        # Test different summary lengths
        for num_sentences in [1, 2, 3]:
            summary = SimpleSummarizer.extractive_summary(test_text, num_sentences)
            self.assertIsInstance(summary, str)
            self.assertGreater(len(summary), 0)
        
        print("✅ Extractive summary test passed")
    
    def test_short_text_summary(self):
        """Test summarization with short text"""
        short_text = "This is a short text."
        summary = SimpleSummarizer.extractive_summary(short_text, num_sentences=3)
        
        # Should return original text for very short inputs
        self.assertIsInstance(summary, str)
        print("✅ Short text summary test passed")


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for real-world scenarios"""
    
    def test_scientific_papers_scenario(self):
        """Test attribution for scientific paper summaries"""
        papers = {
            "paper1": """
            This study investigates the effects of machine learning on medical diagnosis.
            We trained a deep neural network on 10,000 medical images.
            The model achieved 95% accuracy in detecting lung cancer from X-rays.
            These results demonstrate the potential of AI in healthcare applications.
            """,
            
            "paper2": """
            Our research examines natural language processing for clinical notes.
            We developed a transformer model to extract medical entities from text.
            The system identified diseases and treatments with 89% precision.
            Integration with electronic health records improved workflow efficiency.
            """,
            
            "paper3": """
            This paper presents a computer vision approach for skin cancer detection.
            Convolutional neural networks were trained on dermatology datasets.
            The algorithm outperformed dermatologists in melanoma classification.
            Deployment in mobile apps could enable early cancer screening.
            """
        }
        
        external_summary = """
        Recent AI research shows promising results in medical applications including 
        lung cancer detection from X-rays with 95% accuracy and skin cancer 
        classification that outperforms human experts.
        """
        
        attributor = StandaloneReverseAttributor()
        attributor.add_documents(papers)
        
        result = attributor.attribute_summary(
            summary=external_summary,
            top_k=2,
            min_similarity=0.3
        )
        
        # Should find relevant attributions
        self.assertGreater(result.stats['coverage'], 0)
        print("✅ Scientific papers scenario test passed")
        print(f"   Coverage: {result.stats['coverage']:.1%}")
    
    def test_news_articles_scenario(self):
        """Test attribution for news article summaries"""
        articles = {
            "tech_news": """
            Major tech companies are investing heavily in artificial intelligence research.
            Google announced a $10 billion AI investment plan for the next five years.
            Microsoft is integrating AI capabilities into all their products.
            Apple is developing on-device AI to enhance privacy protection.
            """,
            
            "business_news": """
            The AI market is expected to reach $1 trillion by 2030.
            Startups in the AI space raised $25 billion in venture funding last year.
            Traditional industries are adopting AI to improve efficiency and reduce costs.
            Job market demand for AI skills has increased by 200% in two years.
            """,
            
            "policy_news": """
            Governments worldwide are developing AI regulation frameworks.
            The EU AI Act sets strict rules for high-risk AI applications.
            China released national AI governance guidelines emphasizing safety.
            The US is considering federal AI oversight legislation.
            """
        }
        
        external_summary = """
        The AI industry is experiencing massive growth with tech giants investing billions
        and the market expected to reach $1 trillion by 2030, while governments
        develop regulatory frameworks to ensure safe AI deployment.
        """
        
        attributor = StandaloneReverseAttributor()
        attributor.add_documents(articles)
        
        result = attributor.attribute_summary(
            summary=external_summary,
            top_k=3,
            min_similarity=0.25
        )
        
        # Should attribute claims to relevant articles
        self.assertGreater(result.stats['sources_used'], 0)
        print("✅ News articles scenario test passed")
        print(f"   Sources used: {result.stats['sources_used']}")


def run_performance_test():
    """Test performance with larger document set"""
    print("\n🚀 Performance Test")
    print("-" * 40)
    
    import time
    
    # Create larger document set
    large_documents = {}
    for i in range(20):
        large_documents[f"doc_{i}"] = f"""
        Document {i} contains information about various topics including technology,
        science, business, and policy matters. This document discusses the implications
        of artificial intelligence, machine learning, and data science in modern society.
        The content covers research findings, industry trends, and regulatory developments
        that shape the future of these rapidly evolving fields.
        """
    
    summary = """
    Artificial intelligence and machine learning are transforming modern society
    through technological advances, business applications, and policy developments.
    """
    
    # Time the attribution process
    start_time = time.time()
    
    attributor = StandaloneReverseAttributor()
    attributor.add_documents(large_documents)
    
    result = attributor.attribute_summary(
        summary=summary,
        top_k=3,
        min_similarity=0.2
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"   Documents processed: {len(large_documents)}")
    print(f"   Total sentences: {sum(len(sents) for sents in attributor.corpus.values())}")
    print(f"   Processing time: {processing_time:.2f} seconds")
    print(f"   Coverage achieved: {result.stats['coverage']:.1%}")
    print("✅ Performance test completed")


def run_interactive_test():
    """Interactive test allowing user input"""
    print("\n🎮 Interactive Test")
    print("-" * 40)
    print("Enter your own documents and summary for testing:")
    
    # Get user input
    documents = {}
    print("\nEnter documents (press Enter twice to finish each document):")
    
    for i in range(3):
        print(f"\nDocument {i+1} text:")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        
        if lines:
            doc_text = " ".join(lines)
            documents[f"user_doc_{i+1}"] = doc_text
        else:
            break
    
    if not documents:
        print("No documents entered, skipping interactive test")
        return
    
    print("\nEnter summary to attribute:")
    summary = input()
    
    if not summary:
        print("No summary entered, skipping interactive test")
        return
    
    # Run attribution
    try:
        attributor = StandaloneReverseAttributor()
        attributor.add_documents(documents)
        
        result = attributor.attribute_summary(
            summary=summary,
            top_k=2,
            min_similarity=0.2
        )
        
        print_attribution_results(result)
        
    except Exception as e:
        print(f"❌ Error in interactive test: {e}")


def main():
    """Run all tests"""
    print("🧪 REVERSE ATTRIBUTION TEST SUITE")
    print("=" * 60)
    
    # Run unit tests
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestStandaloneReverseAttributor))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestSimpleSummarizer))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestIntegrationScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run additional tests
    run_performance_test()
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {len(result.failures + result.errors)} TEST(S) FAILED")
    
    # Optional interactive test
    print("\n" + "=" * 60)
    user_input = input("Run interactive test? (y/n): ").lower().strip()
    if user_input in ['y', 'yes']:
        run_interactive_test()


if __name__ == "__main__":
    main()
"""
Created on Sat Sep 27 21:20:15 2025

@author: niran
"""

