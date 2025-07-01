#!/usr/bin/env python3
"""
Test the improved text matching in document viewer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_viewer import highlight_text_in_document

def test_text_matching():
    print("="*60)
    print("🧪 TESTING IMPROVED TEXT MATCHING")
    print("="*60)
    
    # Sample document content (simulated PDF content)
    document_content = """The Importance of Sleep

Chapter 1: Introduction 
Sleep is essential for health and well-being.

The Health Benefits of Sleep
a) Physical Health
Getting enough sleep helps the body regulate vital systems:
- Immune system: Sleep supports immune function.
- Heart health: Helps maintain healthy blood pressure and reduces heart disease risk.
- Metabolism: Impacts how the body processes glucose, linked to weight gain and diabetes.
- Growth and repair: Promotes growth in children and assists cellular repair in adults.

Chapter 2: Sleep Deprivation Effects
irregular schedules, shift work, and stress all contribute to widespread sleep deprivation.
The Consequences of Poor Sleep
Sleep deprivation isn't just about feeling tired - it has serious consequences:
- Cognitive Impairment: Affects concentration, judgment, and decision-making skills.
- Emotional Instability: More likely to experience mood disorders.
- Increased Risk of Accidents: Sleepy driving is as dangerous as drunk driving."""
    
    # Test chunks with different formatting issues
    test_chunks = [
        {
            'text': 'The Health Benefits of Sleep\na) Physical Health\nGetting enough sleep helps the body regulate vital systems:\n- Immune system: Sleep supports immune function.',
            'score': 0.8
        },
        {
            'text': 'Cognitive Impairment: Affects concentration, judgment, and decision-making skills.',
            'score': 0.9
        },
        {
            'text': 'Sleep is essential for health and well-being.',  # Should match exactly
            'score': 0.7
        }
    ]
    
    print(f"📄 Testing with document length: {len(document_content)} chars")
    print(f"📦 Testing with {len(test_chunks)} chunks")
    
    # Save content to temp file for testing
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(document_content)
        temp_path = f.name
    
    try:
        # Test the highlighting
        result = highlight_text_in_document(temp_path, test_chunks)
        
        # Check if highlighting was applied
        if '<mark' in result:
            print("✅ SUCCESS: Highlighting was applied!")
            
            # Count how many chunks were highlighted
            highlight_count = result.count('<mark')
            print(f"📊 Total highlights applied: {highlight_count}")
            
            # Show sample of highlighted content
            if 'Cognitive Impairment' in result and '<mark' in result:
                start_idx = result.find('Cognitive Impairment') - 50
                end_idx = result.find('Cognitive Impairment') + 150
                sample = result[max(0, start_idx):end_idx]
                print(f"💡 Sample highlight: ...{sample}...")
        else:
            print("❌ FAILED: No highlighting was applied")
            print(f"Result preview: {result[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        return False
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

if __name__ == "__main__":
    success = test_text_matching()
    print(f"\n{'✅ TEST PASSED' if success else '❌ TEST FAILED'}")
    print("\n💡 Next: Test with the actual chatbot!")
    print("   1. Run: streamlit run chatbot_app.py")
    print("   2. Upload: The_Importance_of_Sleep (1).pdf")  
    print("   3. Ask: 'What are some physical health benefits of sleep?'")
    print("   4. Check if the full health benefits section is highlighted")
