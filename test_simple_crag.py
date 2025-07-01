#!/usr/bin/env python3
"""
Simple test to verify CRAG improvements with the sleep PDF
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crag import CRAG

def test_with_actual_pdf():
    print("="*60)
    print("🔧 TESTING CRAG IMPROVEMENTS WITH ACTUAL PDF")
    print("="*60)
    
    # Use the actual sleep PDF from Downloads
    pdf_path = r"C:\Users\iTECH\AppData\Local\Temp\tmp_d4qb_ip\The_Importance_of_Sleep (1).pdf"
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found at: {pdf_path}")
        print("Please ensure the PDF is available")
        return False
    
    query = "What are two cognitive effects of sleep deprivation?"
    
    print(f"📄 Testing with: {os.path.basename(pdf_path)}")
    print(f"🔍 Query: {query}")
    
    try:
        # Initialize CRAG with stricter evaluation
        crag = CRAG(
            file_path=pdf_path,
            lower_threshold=0.3,
            upper_threshold=0.7
        )
        
        print("\n🤖 Running CRAG with improved scoring...")
        result = crag.run_with_sources(query)
        
        print(f"\n📊 CRAG Result:")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources count: {len(result.get('sources', []))}")
        
        # Analyze the source chunks
        sources = result.get('sources', [])
        print(f"\n📋 Source Analysis:")
        
        for i, source in enumerate(sources):
            print(f"\nSource {i+1}:")
            print(f"  Score: {source.get('score', 'N/A')}")
            print(f"  Page: {source.get('page', 'N/A')}")
            print(f"  Text preview: {source.get('text', '')[:100]}...")
            
            # Check for problematic patterns
            text_lower = source.get('text', '').lower()
            
            if "health benefits" in text_lower and "cognitive" not in text_lower:
                score = source.get('score', 0)
                print(f"  ⚠️  GENERIC HEALTH BENEFITS chunk detected!")
                if score > 0.5:
                    print(f"  ❌ ERROR: Generic health chunk has high score ({score})")
                else:
                    print(f"  ✅ GOOD: Generic health chunk has low score ({score})")
            
            if "cognitive" in text_lower and any(word in text_lower for word in ["concentration", "judgment", "decision", "attention"]):
                print(f"  ✅ RELEVANT: Directly answers cognitive effects question")
        
        print(f"\n🎯 Test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_with_actual_pdf()
