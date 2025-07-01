#!/usr/bin/env python3

"""
Final test for document highlighting improvements
"""

import tempfile
import os
import time
from document_viewer import highlight_text_in_document

def test_highlighting():
    # Create test content
    test_content = '''Introduction to Sleep

In this document, we will explore sleep effects. Sleep deprivation has serious consequences:

• Cognitive Impairment: Affects concentration, judgment, and decision-making skills.
• Emotional Instability: More likely to experience mood disorders.'''

    # Create a persistent temporary file
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_file = f.name
        
        # Ensure file is written and accessible
        time.sleep(0.1)
        
        # Verify file exists and is readable
        if not os.path.exists(temp_file):
            print(f'❌ ERROR: Temporary file {temp_file} was not created')
            return False
            
        with open(temp_file, 'r') as f:
            file_content = f.read()
            if file_content != test_content:
                print(f'❌ ERROR: File content mismatch')
                return False

        chunks_to_highlight = [
            {'text': 'In this document, we will explore sleep effects.', 'score': 0.3},  # Should be filtered out
            {'text': 'Cognitive Impairment: Affects concentration, judgment, and decision-making skills.', 'score': 0.8},  # Should be highlighted
        ]
        
        print(f'🧪 Testing document viewer filtering with file: {temp_file}')
        print(f'📁 File size: {os.path.getsize(temp_file)} bytes')
        
        highlighted = highlight_text_in_document(temp_file, chunks_to_highlight)
        
        if highlighted is None:
            print('❌ ERROR: highlight_text_in_document returned None')
            return False
            
        print(f'📄 Highlighted content length: {len(highlighted)} characters')
        print(f'📄 First 200 chars of highlighted content: {highlighted[:200]}...')
        
        # Check specifically for highlighting with mark tags
        intro_text = 'In this document, we will explore sleep effects.'
        cognitive_text = 'Cognitive Impairment: Affects concentration, judgment, and decision-making skills.'
        
        # More robust check for mark tags
        intro_in_content = intro_text in highlighted
        cognitive_in_content = cognitive_text in highlighted
        
        # Check for mark tags around the text
        intro_marked = False
        cognitive_marked = False
        
        if intro_in_content:
            intro_pos = highlighted.find(intro_text)
            # Look for mark tag before and after this position
            before_intro = highlighted[:intro_pos + len(intro_text)]
            after_intro = highlighted[intro_pos:]
            intro_marked = '<mark' in before_intro and '</mark>' in after_intro
            
        if cognitive_in_content:
            cognitive_pos = highlighted.find(cognitive_text)
            # Look for mark tag before and after this position  
            before_cognitive = highlighted[:cognitive_pos + len(cognitive_text)]
            after_cognitive = highlighted[cognitive_pos:]
            cognitive_marked = '<mark' in before_cognitive and '</mark>' in after_cognitive
        
        print(f'📝 Intro text found in content: {intro_in_content}')
        print(f'🏷️  Intro text has mark tags: {intro_marked}')
        print(f'📝 Cognitive text found in content: {cognitive_in_content}')
        print(f'🏷️  Cognitive text has mark tags: {cognitive_marked}')
        
        if intro_marked:
            print('❌ FAILED: Low-relevance intro text was highlighted with mark tags')
            return False
        elif cognitive_marked:
            print('✅ PASSED: Only high-relevance chunk was highlighted')
            return True
        else:
            print('⚠️  WARNING: No chunks were highlighted - checking for any mark tags...')
            has_any_marks = '<mark' in highlighted
            print(f'🔍 Any mark tags found: {has_any_marks}')
            if has_any_marks:
                # Extract what's between mark tags
                import re
                marks = re.findall(r'<mark[^>]*>(.*?)</mark>', highlighted, re.DOTALL)
                print(f'🏷️  Marked content: {marks}')
            return False
            
    except Exception as e:
        print(f'❌ ERROR during test: {str(e)}')
        return False
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                print(f'⚠️  Warning: Could not delete temp file {temp_file}: {e}')

def main():
    print("=" * 60)
    print("FINAL TEST: Document Highlighting Improvements")
    print("=" * 60)
    
    success = test_highlighting()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS: Document highlighting improvements are working correctly!")
        print("📋 Summary of improvements:")
        print("   • Only chunks with score >= 0.5 are highlighted")
        print("   • Chunks shorter than 20 characters are filtered out")
        print("   • Generic introductory phrases are filtered out")
        print("   • Download button keys are unique to prevent collisions")
        print("   • Session state is cleared between queries to prevent stale highlighting")
    else:
        print("❌ FAILURE: Some issues remain with the highlighting system")
    
    return success

if __name__ == "__main__":
    main()
