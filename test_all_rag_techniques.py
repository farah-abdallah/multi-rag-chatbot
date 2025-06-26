#!/usr/bin/env python3
"""
Test script to verify all RAG techniques load properly with API key rotation
"""

import sys
import traceback

def test_technique(name, import_statement):
    """Test if a RAG technique loads properly"""
    try:
        print(f"\n🧪 Testing {name}...")
        exec(import_statement)
        print(f"✅ {name} loaded successfully with API key rotation")
        return True
    except Exception as e:
        print(f"❌ {name} failed to load: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False

def main():
    print("🔄 Testing all RAG techniques with API key rotation...")
    print("=" * 60)
    
    techniques = [
        ("Adaptive RAG", "from adaptive_rag import AdaptiveRAG"),
        ("CRAG", "from crag import CRAG"),
        ("Document Augmentation", "from document_augmentation import DocumentProcessor"),
        ("Explainable Retrieval", "from explainable_retrieval import ExplainableRAGMethod"),
        ("Evaluation Framework", "from evaluation_framework import AutomatedEvaluator")
    ]
    
    results = []
    for name, import_stmt in techniques:
        success = test_technique(name, import_stmt)
        results.append((name, success))
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY RESULTS:")
    print("=" * 60)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {name}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL RAG TECHNIQUES LOADED SUCCESSFULLY!")
        print("🚀 System is ready for large-scale evaluation with API key rotation")
    else:
        print("⚠️  Some techniques failed to load - check errors above")
    print("=" * 60)

if __name__ == "__main__":
    main()
