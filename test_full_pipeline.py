#!/usr/bin/env python3

"""
Test the full CRAG + Document Viewer pipeline for mental health query
"""

import tempfile
import os
from crag import CRAG
from document_viewer import highlight_text_in_document

def test_full_pipeline():
    print("============================================================")
    print("FULL PIPELINE TEST: CRAG + Document Viewer")
    print("============================================================")
    
    # Create test content - same as user's document
    test_content = '''The Health Benefits of Sleep

a) Physical Health
Getting enough sleep helps the body regulate vital systems, supports immune function, and helps maintain cardiovascular health. Sleep also aids in tissue repair and growth hormone release.

b) Mental and Emotional Health
Sleep improves cognitive functions like attention, problem-solving, and creativity. It also processes information and forms memories, and improves emotional regulation, reducing mood swings, anxiety, and depression risk.'''

    # Create temporary file
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file = f.name
        
        print(f"✅ Created test document: {temp_file}")
        print(f"📄 Content preview: {test_content[:100]}...")
        
        # Initialize CRAG
        print("\n🔧 Initializing CRAG...")
        crag = CRAG(temp_file, web_search_enabled=False)
        
        # Test the mental health query
        query = "How does sleep support mental and emotional health?"
        print(f"\n🎯 Query: {query}")
        
        # Run CRAG with source tracking
        print("\n🔍 Running CRAG with source tracking...")
        result = crag.run_with_sources(query)
        
        print(f"\n📝 CRAG Answer:")
        print(f"{result['answer'][:300]}...")
        
        print(f"\n📋 Source chunks from CRAG:")
        source_chunks = result['source_chunks']
        print(f"   Total chunks: {len(source_chunks)}")
        
        for i, chunk in enumerate(source_chunks):
            score = chunk.get('score', 'N/A')
            text = chunk.get('text', '')[:100] + '...' if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
            print(f"   {i+1}. Score: {score}, Text: '{text}'")
        
        # Test document viewer highlighting
        print(f"\n🎨 Testing document viewer highlighting...")
        
        # Convert CRAG chunks to document viewer format
        chunks_for_highlighting = []
        for chunk in source_chunks:
            chunks_for_highlighting.append({
                'text': chunk.get('text', ''),
                'score': chunk.get('score', 0.0)
            })
        
        print(f"📦 Chunks to highlight: {len(chunks_for_highlighting)}")
        for i, chunk in enumerate(chunks_for_highlighting):
            text_preview = chunk['text'][:50] + '...' if len(chunk['text']) > 50 else chunk['text']
            print(f"   {i+1}. Score: {chunk['score']}, Text: '{text_preview}'")
        
        # Highlight in document
        highlighted_content = highlight_text_in_document(temp_file, chunks_for_highlighting)
        
        # Check if mental health content is highlighted
        mental_text = "Sleep improves cognitive functions like attention, problem-solving, and creativity"
        physical_text = "Getting enough sleep helps the body regulate vital systems"
        
        mental_highlighted = '<mark' in highlighted_content and mental_text in highlighted_content
        physical_highlighted = '<mark' in highlighted_content and physical_text in highlighted_content
        
        print(f"\n🔍 Highlighting Analysis:")
        print(f"   Mental health text found: {mental_text in highlighted_content}")
        print(f"   Mental health text highlighted: {mental_highlighted}")
        print(f"   Physical health text found: {physical_text in highlighted_content}")
        print(f"   Physical health text highlighted: {physical_highlighted}")
        
        # Show what's actually highlighted
        if '<mark' in highlighted_content:
            import re
            marked_content = re.findall(r'<mark[^>]*>(.*?)</mark>', highlighted_content, re.DOTALL)
            print(f"\n🏷️  Actually highlighted content:")
            for i, content in enumerate(marked_content):
                content_clean = content.strip()[:100] + '...' if len(content.strip()) > 100 else content.strip()
                print(f"   {i+1}. '{content_clean}'")
        else:
            print(f"\n❌ No content is highlighted!")
        
        # Final assessment
        if mental_highlighted and not physical_highlighted:
            print(f"\n✅ SUCCESS: Only mental/emotional health content is highlighted!")
            return True
        elif mental_highlighted and physical_highlighted:
            print(f"\n⚠️  WARNING: Both mental and physical content are highlighted")
            return False
        elif physical_highlighted and not mental_highlighted:
            print(f"\n❌ FAILURE: Only physical health content is highlighted (wrong content!)")
            return False
        else:
            print(f"\n❌ FAILURE: No relevant content is highlighted")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                print(f"⚠️  Warning: Could not delete temp file: {e}")

def main():
    success = test_full_pipeline()
    
    print("\n" + "="*60)
    if success:
        print("🎉 SUCCESS: Full pipeline correctly highlights mental/emotional health content!")
    else:
        print("❌ FAILURE: Pipeline needs further debugging")
    
    return success

if __name__ == "__main__":
    main()
