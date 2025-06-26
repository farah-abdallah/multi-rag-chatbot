#!/usr/bin/env python3
"""
Demonstration: How to use AdaptiveRAG with silent mode for proper faithfulness evaluation
"""

import adaptive_rag

def demonstrate_proper_evaluation():
    """
    Shows the correct way to call AdaptiveRAG for faithfulness evaluation
    """
    print("🎯 Demonstrating Proper AdaptiveRAG Evaluation")
    print("="*60)
    
    # Create test data
    texts = [
        'Climate change refers to long-term shifts in global temperatures and weather patterns.',
        'The main cause of climate change is the emission of greenhouse gases like CO2.',
        'Effects of climate change include rising sea levels and extreme weather events.',
        'Renewable energy sources can help reduce greenhouse gas emissions.',
        'Fossil fuels like coal and oil contribute significantly to greenhouse gas emissions.'
    ]
    
    # Initialize AdaptiveRAG
    print("1. Creating AdaptiveRAG instance...")
    rag = adaptive_rag.AdaptiveRAG(texts=texts)
    
    # Test queries
    queries = [
        "What causes climate change?",
        "What are the effects of climate change?", 
        "How can we reduce emissions?"
    ]
    
    print("\n2. Testing queries with SILENT mode (for evaluation)...")
    print("-" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        
        # CRITICAL: Use silent=True for evaluation to avoid debug output
        context = rag.get_context_for_query(query, silent=True)
        answer = rag.answer(query, silent=True)
        
        print(f"   Context length: {len(context)} chars")
        print(f"   Answer length: {len(answer)} chars")
        print(f"   Context: {context[:80]}...")
        print(f"   Answer: {answer[:80]}...")
        
        # Simple faithfulness check (this is what your evaluation should do)
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(context_words.intersection(answer_words))
        faithfulness = min(overlap / max(len(answer_words), 1), 1.0)
        
        print(f"   📊 Estimated faithfulness: {faithfulness:.3f}")
    
    print("\n" + "="*60)
    print("✅ KEY POINTS FOR YOUR EVALUATION FRAMEWORK:")
    print("   1. Always call: rag.get_context_for_query(query, silent=True)")
    print("   2. Always call: rag.answer(query, silent=True)")
    print("   3. Use the SAME context from get_context_for_query() for faithfulness")
    print("   4. DO NOT reconstruct context separately!")
    print("\n💡 This should fix the 0.5 faithfulness score issue!")

if __name__ == "__main__":
    demonstrate_proper_evaluation()
