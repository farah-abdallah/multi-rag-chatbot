#!/usr/bin/env python3
"""
Simple focused test for highlighting fix
"""

import sys
import os
import tempfile
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_viewer import highlight_text_in_document

def test_highlighting_fix():
    print("="*60)
    print("🎯 FOCUSED HIGHLIGHTING FIX TEST")
    print("="*60)
    
    # Simple document content
    document_content = """The Health Benefits of Sleep
a) Physical Health
Getting enough sleep helps the body regulate vital systems:
- Immune system: Sleep supports immune function.
- Heart health: Helps maintain healthy blood pressure and reduces heart disease risk.
- Metabolism: Impacts how the body processes glucose, linked to weight gain and diabetes.

The Consequences of Poor Sleep
Sleep deprivation has serious consequences:
- Cognitive Impairment: Affects concentration, judgment, and decision-making skills.
- Emotional Instability: More likely to experience mood disorders."""
    
    # Test chunk (similar to what CRAG provides)
    test_chunk = {
        'text': """The Health Benefits of Sleep
a) Physical Health
Getting enough sleep helps the body regulate vital systems:
- Immune system: Sleep supports immune function.
- Heart health: Helps maintain healthy blood pressure and reduces heart disease risk.
- Metabolism: Impacts how the body processes glucose, linked to weight gain and diabetes.""",
        'score': 0.8
    }
    
    print(f"📄 Document content length: {len(document_content)} chars")
    print(f"📦 Chunk content length: {len(test_chunk['text'])} chars")
    print(f"📊 Chunk score: {test_chunk['score']}")
    
    # Save to a proper temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(document_content)
        temp_path = f.name
    
    try:
        print(f"💾 Saved test content to: {temp_path}")
        
        # Test highlighting
        result = highlight_text_in_document(temp_path, [test_chunk])
        
        print(f"\n📊 RESULTS:")
        print(f"Result length: {len(result)} chars")
        
        # Check if highlighting was applied
        if '<mark' in result:
            highlight_count = result.count('<mark')
            print(f"✅ SUCCESS: {highlight_count} highlights applied!")
            
            # Show where highlights are
            lines = result.split('\n')
            for i, line in enumerate(lines):
                if '<mark' in line:
                    print(f"  Line {i+1}: {line}")
        else:
            print("❌ FAILED: No highlighting applied")
            print(f"Result preview: {result[:300]}...")
            
            # Debug: Check if the text exists at all
            if "Health Benefits" in result:
                print("✅ Text exists in document")
            else:
                print("❌ Text missing from document")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean up
        try:
            os.unlink(temp_path)
        except:
            pass

if __name__ == "__main__":
    test_highlighting_fix()
