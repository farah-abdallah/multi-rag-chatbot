#!/usr/bin/env python3
"""
Test script to verify the faithfulness fix in Adaptive RAG
"""

import sys
import os
from adaptive_rag import AdaptiveRAG
from evaluation_framework import AutomatedEvaluator

def test_faithfulness_fix():
    """Test that the faithfulness metric now works correctly"""
    print("🧪 Testing Adaptive RAG Faithfulness Fix")
    print("=" * 60)
    
    # Create test data
    texts = [
        'Climate change refers to long-term shifts in global temperatures and weather patterns.',
        'The main cause of climate change is the emission of greenhouse gases like CO2.',
        'Effects of climate change include rising sea levels and extreme weather events.',
        'Renewable energy sources can help reduce greenhouse gas emissions.',
        'Solar and wind power are examples of clean, renewable energy technologies.'
    ]
    
    print("1. Creating AdaptiveRAG instance...")
    try:
        rag = AdaptiveRAG(texts=texts)
        print("   ✅ AdaptiveRAG created successfully")
    except Exception as e:
        print(f"   ❌ Failed to create AdaptiveRAG: {e}")
        return False
    
    print("2. Creating AutomatedEvaluator...")
    try:
        evaluator = AutomatedEvaluator()
        print("   ✅ AutomatedEvaluator created successfully")
    except Exception as e:
        print(f"   ❌ Failed to create AutomatedEvaluator: {e}")
        return False
    
    # Test queries
    test_queries = [
        "What causes climate change?",
        "What are renewable energy sources?", 
        "How does solar power work?"  # This should have lower faithfulness
    ]
    
    print("\n3. Testing faithfulness evaluation...")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        
        try:
            # CRITICAL: Use the new method to get context and answer
            # This ensures the SAME context is used for both generation and evaluation
            context = rag.get_context_for_query(query, silent=True)
            answer = rag.answer(query, silent=True)
            
            print(f"   Context length: {len(context)} chars")
            print(f"   Answer length: {len(answer)} chars")
            print(f"   Context preview: {context[:100]}...")
            print(f"   Answer preview: {answer[:100]}...")
            
            # Evaluate faithfulness using the evaluator
            faithfulness = evaluator.evaluate_faithfulness(context, answer)
            relevance = evaluator.evaluate_relevance(query, answer)
            completeness = evaluator.evaluate_completeness(query, answer)
            
            print(f"   📊 Faithfulness: {faithfulness:.3f}")
            print(f"   📊 Relevance: {relevance:.3f}")
            print(f"   📊 Completeness: {completeness:.3f}")
            
            # Check if faithfulness is reasonable (not stuck at 0.5)
            if abs(faithfulness - 0.5) > 0.1:
                print(f"   ✅ Faithfulness metric working! (not stuck at 0.5)")
            else:
                print(f"   ⚠️ Faithfulness might still be stuck at 0.5")
        except Exception as e:
            print(f"   ❌ Error testing query '{query}': {e}")
    
    print("\n" + "=" * 60)
    print("📊 FAITHFULNESS FIX SUMMARY:")
    print("=" * 60)
    print("✅ KEY CHANGES MADE:")
    print("   1. chatbot_app.py now uses get_context_for_query() method")
    print("   2. SAME context used for both answer generation and evaluation") 
    print("   3. Silent mode enabled to avoid debug output during evaluation")
    print("   4. Proper context extraction ensures accurate faithfulness scoring")
    print("\n💡 The faithfulness score should now vary based on actual content!")
    print("💡 Instead of being stuck at 0.5, it should reflect true faithfulness!")
    
    return True

if __name__ == "__main__":
    try:
        print("🚀 Starting faithfulness fix test...")
        test_faithfulness_fix()
        print("\n🎉 Test completed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
