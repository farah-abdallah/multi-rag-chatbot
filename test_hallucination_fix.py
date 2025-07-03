#!/usr/bin/env python3
"""
Test script to verify that hallucination fixes work correctly
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crag import CRAG

def test_no_hallucination():
    """Test that responses stick to source material only"""
    print("🧪 TESTING HALLUCINATION PREVENTION")
    print("=" * 60)
    
    # Test with sleep document if available
    test_docs = [
        "The_Importance_of_Sleep (1).pdf",
        "data/Understanding_Climate_Change (1).pdf"
    ]
    
    test_doc = None
    for doc in test_docs:
        if os.path.exists(doc):
            test_doc = doc
            break
    
    if not test_doc:
        print("❌ No test documents found. Looking for any available PDF...")
        for file in os.listdir("."):
            if file.endswith(".pdf"):
                test_doc = file
                break
    
    if not test_doc:
        print("❌ No PDF documents found. Exiting test.")
        return
    
    print(f"✅ Using test document: {test_doc}")
    
    try:
        # Initialize CRAG with strict source validation
        crag = CRAG(
            file_path=test_doc,
            model="gemini-1.5-flash",
            web_search_enabled=False,  # Disable web search to test source-only responses
            lower_threshold=0.3,
            upper_threshold=0.7
        )
        
        # Test questions that might trigger hallucination
        test_questions = [
            "Why should exercise timing be considered in relation to sleep?",
            "What are the health benefits of sleep?",
            "How much sleep do adults need?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*50}")
            print(f"TEST {i}: {question}")
            print(f"{'='*50}")
            
            # Get response with source tracking
            result = crag.run_with_sources(question)
            answer = result['answer']
            source_chunks = result['source_chunks']
            
            print(f"\n📝 ANSWER:")
            print(answer)
            
            print(f"\n📊 ANALYSIS:")
            print(f"   Answer length: {len(answer.split())} words")
            print(f"   Source chunks used: {len(source_chunks)}")
            
            if source_chunks:
                total_source_words = sum(len(chunk.get('text', '').split()) for chunk in source_chunks)
                print(f"   Total source material: {total_source_words} words")
                
                # Check ratio
                ratio = len(answer.split()) / total_source_words if total_source_words > 0 else 0
                print(f"   Answer/Source ratio: {ratio:.2f}")
                
                if ratio > 0.6:
                    print("   ⚠️ WARNING: Answer may be too verbose relative to sources")
                elif ratio < 0.3:
                    print("   ✅ GOOD: Concise answer relative to sources")
                else:
                    print("   ✅ GOOD: Reasonable answer length")
            
            # Check for common hallucination patterns
            hallucination_phrases = [
                "research shows that", "studies indicate", "it is well known",
                "physiological processes", "elevated energy levels", "body temperature disrupts"
            ]
            
            found_phrases = [phrase for phrase in hallucination_phrases if phrase in answer.lower()]
            if found_phrases:
                print(f"   ⚠️ Potential hallucination indicators: {found_phrases}")
                
                # Check if these phrases exist in source material
                source_text = " ".join([chunk.get('text', '') for chunk in source_chunks])
                for phrase in found_phrases:
                    if phrase not in source_text.lower():
                        print(f"   ❌ '{phrase}' NOT found in source material - possible hallucination")
                    else:
                        print(f"   ✅ '{phrase}' found in source material")
            else:
                print("   ✅ No obvious hallucination indicators detected")
            
            print("\n" + "-"*50)
        
        print(f"\n🎯 SUMMARY:")
        print("✅ Hallucination prevention test completed")
        print("✅ Check warnings above for potential issues")
        print("✅ Responses should now be more source-constrained and concise")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_no_hallucination()
