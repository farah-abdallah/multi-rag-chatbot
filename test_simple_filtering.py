#!/usr/bin/env python3

"""
Simple verification of chunk filtering improvements
"""

import tempfile
import os

def test_simple_highlighting():
    """Test chunk highlighting with simpler logic"""
    
    # Create test content
    test_content = """Introduction to Sleep

In this document, we will explore sleep effects. Sleep deprivation has serious consequences:

• Cognitive Impairment: Affects concentration, judgment, and decision-making skills.
• Emotional Instability: More likely to experience mood disorders."""

    # Write content to a regular file (not temp file)
    test_file = "test_document.txt"
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Test chunks
        chunks_to_highlight = [
            {'text': 'In this document, we will explore sleep effects.', 'score': 0.3},  # Should be filtered out (low score)
            {'text': 'Cognitive Impairment: Affects concentration, judgment, and decision-making skills.', 'score': 0.8},  # Should be highlighted
            {'text': 'Short', 'score': 0.9},  # Should be filtered out (too short)
        ]
        
        print("=== Testing Chunk Filtering Logic ===")
        print("Content:")
        print(test_content)
        print("\nTest chunks:")
        for i, chunk in enumerate(chunks_to_highlight):
            print(f"  {i+1}. Score: {chunk['score']}, Text: '{chunk['text']}'")
        
        # Apply filtering logic manually to verify
        print("\n=== Applying Filters ===")
        
        # Score filter (>= 0.5)
        score_filtered = [chunk for chunk in chunks_to_highlight if chunk.get('score', 0) >= 0.5]
        print(f"After score filter (>= 0.5): {len(score_filtered)} chunks")
        for chunk in score_filtered:
            print(f"  - Score: {chunk['score']}, Text: '{chunk['text'][:50]}...'")
        
        # Length filter (>= 20 chars)
        length_filtered = [chunk for chunk in score_filtered if len(chunk.get('text', '')) >= 20]
        print(f"After length filter (>= 20 chars): {len(length_filtered)} chunks")
        for chunk in length_filtered:
            print(f"  - Length: {len(chunk['text'])}, Text: '{chunk['text'][:50]}...'")
        
        # Generic phrase filter
        generic_phrases = ["in this document", "we will explore", "we will examine"]
        final_filtered = []
        for chunk in length_filtered:
            chunk_text = chunk.get('text', '').lower()
            has_generic = any(phrase in chunk_text for phrase in generic_phrases)
            is_short = len(chunk_text) < 100
            
            if has_generic and is_short:
                print(f"  - FILTERED OUT (generic + short): '{chunk['text'][:50]}...'")
            else:
                final_filtered.append(chunk)
                print(f"  - KEPT: '{chunk['text'][:50]}...'")
        
        print(f"\nFinal chunks to highlight: {len(final_filtered)}")
        
        # Test the actual highlighting
        print("\n=== Testing Document Viewer Function ===")
        try:
            from document_viewer import highlight_text_in_document
            
            highlighted = highlight_text_in_document(test_file, chunks_to_highlight)
            
            # Check if highlighting worked
            has_marks = '<mark' in highlighted
            cognitive_highlighted = 'Cognitive Impairment' in highlighted and '<mark' in highlighted
            intro_highlighted = ('In this document' in highlighted and 
                               highlighted.find('<mark') < highlighted.find('In this document') < 
                               highlighted.find('</mark>') if '<mark' in highlighted else False)
            
            print(f"Document has highlight marks: {has_marks}")
            print(f"Cognitive text highlighted: {cognitive_highlighted}")
            print(f"Intro text highlighted: {intro_highlighted}")
            
            if cognitive_highlighted and not intro_highlighted:
                print("✅ SUCCESS: Only relevant chunks highlighted!")
            elif intro_highlighted:
                print("❌ FAILED: Low-relevance intro text was highlighted")
            elif not has_marks:
                print("⚠️ WARNING: No text was highlighted")
            else:
                print("❓ UNCLEAR: Unexpected highlighting pattern")
                
        except Exception as e:
            print(f"❌ Error testing document viewer: {e}")
            
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    test_simple_highlighting()
