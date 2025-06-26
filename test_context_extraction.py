#!/usr/bin/env python3
"""
Test script to debug Adaptive RAG context extraction
"""

import adaptive_rag

def test_adaptive_rag_context():
    print("🧪 Testing Adaptive RAG Context Extraction")
    print("="*50)
    
    # Create simple test data
    texts = [
        'Climate change refers to long-term shifts in global temperatures and weather patterns.',
        'The main cause of climate change is the emission of greenhouse gases like CO2.',
        'Effects of climate change include rising sea levels and extreme weather events.',
        'Renewable energy sources can help reduce greenhouse gas emissions.',
        'The Paris Agreement aims to limit global warming to below 2 degrees Celsius.'
    ]
    
    try:
        print(f"📚 Creating AdaptiveRAG with {len(texts)} text chunks...")
        rag = adaptive_rag.AdaptiveRAG(texts=texts)
        
        print("\n🔍 Testing context extraction...")
        test_query = "What causes climate change?"
        
        # Test context extraction method
        print(f"\n1. Getting context for query: '{test_query}'")
        context = rag.get_context_for_query(test_query)
        
        print(f"\n2. Context extracted:")
        print(f"   Length: {len(context)} characters")
        print(f"   Preview: {context[:200]}...")
        
        # Test answer method
        print(f"\n3. Getting answer for same query...")
        answer = rag.answer(test_query)
        
        print(f"\n4. Answer generated:")
        print(f"   Length: {len(answer)} characters")
        print(f"   Preview: {answer[:200]}...")
        
        # Check if contexts match
        print(f"\n5. Context consistency check:")
        print(f"   Last context length: {len(rag.last_context or '')}")
        print(f"   Contexts match: {context == rag.last_context}")
        
        # Test the built-in test method
        print(f"\n6. Running built-in test method...")
        test_results = rag.test_context_extraction(test_query)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_adaptive_rag_context()
    print(f"\n{'✅ Test completed successfully!' if success else '❌ Test failed!'}")
