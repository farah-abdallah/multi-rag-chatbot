#!/usr/bin/env python3

"""
Test CRAG with debugging output
"""

import tempfile
import os

def test_crag_with_debug():
    """Test CRAG with debugging to see what chunks are being highlighted"""
    
    # Create test content similar to your sleep document
    test_content = """Introduction to Sleep

In this document, we will explore sleep effects. Sleep deprivation has serious consequences:

• Cognitive Impairment: Affects concentration, judgment, and decision-making skills.
• Emotional Instability: More likely to experience mood disorders.
• Increased Risk of Accidents: Sleepy driving is as dangerous as drunk driving.
• Chronic Illnesses: Associated with heart disease, stroke, obesity, and certain cancers.

Modern Lifestyle Factors
Unfortunately, modern lifestyles often prevent people from getting enough rest. Social media, irregular schedules, shift work, and stress all contribute to widespread sleep deprivation.

The Consequences of Poor Sleep
Sleep deprivation isn't just about feeling tired - it has serious consequences:"""

    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file = f.name

    try:
        print("🧪 Testing CRAG with debugging...")
        print(f"📄 Test file: {temp_file}")
        
        # Import CRAG
        from crag import CRAG
        
        # Initialize CRAG
        crag = CRAG(temp_file, upper_threshold=0.7, lower_threshold=0.4)
        
        # Test query about cognitive effects (same as your test)
        query = "What are two cognitive effects of sleep deprivation?"
        print(f"\n❓ Query: {query}")
        
        # Run CRAG with sources
        result = crag.run_with_sources(query)
        
        print(f"\n📋 CRAG Results:")
        print(f"   Answer: {result['answer'][:200]}...")
        print(f"   Source chunks returned: {len(result['source_chunks'])}")
        
        # Now test the document viewer
        if result['source_chunks']:
            print(f"\n🖼️  Testing Document Viewer with these chunks...")
            from document_viewer import highlight_text_in_document
            
            highlighted = highlight_text_in_document(temp_file, result['source_chunks'])
            
            # Check if highlighting worked correctly
            has_intro_marks = ('In this document' in highlighted and 
                              highlighted.find('<mark') < highlighted.find('In this document'))
            has_cognitive_marks = ('Cognitive Impairment' in highlighted and 
                                  highlighted.find('<mark') < highlighted.find('Cognitive Impairment'))
            
            print(f"\n🎨 Highlighting Results:")
            print(f"   Intro text highlighted: {has_intro_marks}")
            print(f"   Cognitive text highlighted: {has_cognitive_marks}")
            
            if has_intro_marks:
                print("❌ PROBLEM: Intro text is being highlighted!")
            elif has_cognitive_marks:
                print("✅ GOOD: Only cognitive text is highlighted")
            else:
                print("⚠️  No highlighting detected")
                
        else:
            print("⚠️  No source chunks returned from CRAG")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)

if __name__ == "__main__":
    test_crag_with_debug()
