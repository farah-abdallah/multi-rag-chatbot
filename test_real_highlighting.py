#!/usr/bin/env python3
"""
Final validation test for highlighting functionality - Real world test
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crag import CRAG
from document_viewer import highlight_text_in_document

def main():
    print("🔧 Real-World Highlighting Test")
    print("=" * 50)
    
    # Initialize CRAG
    crag = CRAG()
    
    # Test with sleep document
    pdf_path = "The_Importance_of_Sleep (1).pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    # Test question about sleep benefits
    question = "What are the benefits of sleep for memory and learning?"
    
    print(f"📄 Document: {pdf_path}")
    print(f"❓ Question: {question}")
    print("\n" + "="*60)
    
    # Get answer with sources
    result = crag.run_with_sources(question, pdf_path)
    
    print("\n" + "="*60)
    print("🎯 TESTING DOCUMENT VIEWER HIGHLIGHTING")
    print("="*60)
    
    # Get chunks for highlighting
    chunks_for_highlighting = result.get('source_chunks', [])
    
    print(f"📦 Chunks to highlight: {len(chunks_for_highlighting)}")
    for i, chunk in enumerate(chunks_for_highlighting):
        score = chunk.get('score', 'N/A')
        text_preview = chunk.get('text', '')[:80] + '...' if len(chunk.get('text', '')) > 80 else chunk.get('text', '')
        print(f"  {i+1}. Score: {score}, Text: '{text_preview}'")
    
    # Test highlighting
    print("\n🎨 Calling highlight_text_in_document...")
    highlighted_html = highlight_text_in_document(pdf_path, chunks_for_highlighting)
    
    print("\n" + "="*60)
    print("✅ HIGHLIGHTING TEST COMPLETE")
    print("="*60)
    
    # Check if highlighting was applied
    highlight_count = highlighted_html.count('<mark')
    print(f"🎨 Highlights applied: {highlight_count}")
    
    if highlight_count > 0:
        print("✅ SUCCESS: Text highlighting is working!")
    else:
        print("❌ ISSUE: No highlights were applied")
    
    # Show first few highlights to verify content
    if '<mark' in highlighted_html:
        print("\n📝 Sample highlighted content:")
        import re
        marks = re.findall(r'<mark[^>]*>(.*?)</mark>', highlighted_html, re.DOTALL)
        for i, mark in enumerate(marks[:3]):
            clean_text = mark.strip()[:100] + '...' if len(mark.strip()) > 100 else mark.strip()
            print(f"  Highlight {i+1}: '{clean_text}'")
    
    # Test completeness - check if we're highlighting full sections
    print("\n🔍 Analyzing highlighting completeness:")
    if '<mark' in highlighted_html:
        marks = re.findall(r'<mark[^>]*>(.*?)</mark>', highlighted_html, re.DOTALL)
        total_highlighted_chars = sum(len(mark) for mark in marks)
        total_chunk_chars = sum(len(chunk.get('text', '')) for chunk in chunks_for_highlighting if chunk.get('score', 0) >= 0.5)
        
        print(f"📊 Characters in source chunks: {total_chunk_chars}")
        print(f"📊 Characters highlighted: {total_highlighted_chars}")
        
        if total_highlighted_chars > 0:
            completeness = min(100, (total_highlighted_chars / max(1, total_chunk_chars)) * 100)
            print(f"📊 Highlighting completeness: {completeness:.1f}%")
            
            if completeness > 70:
                print("✅ Good highlighting coverage!")
            elif completeness > 30:
                print("⚠️  Partial highlighting coverage")
            else:
                print("❌ Low highlighting coverage")

if __name__ == "__main__":
    main()
