#!/usr/bin/env python3
"""
Test the fixed page numbers and improved chunk scoring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helper_functions import encode_document
from crag import CRAG
import tempfile

def test_page_number_fix():
    print("="*60)
    print("🔧 TESTING PAGE NUMBER AND CHUNK SCORING FIXES")
    print("="*60)
    
    # Create a simple test PDF content with known pages
    test_content = """The Importance of Sleep

Chapter 1: Introduction 
Sleep is essential for health and well-being.

The Health Benefits of Sleep
a) Physical Health
Getting enough sleep helps the body regulate vital systems:
- Immune system: Sleep supports immune function.

Chapter 2: Sleep Deprivation Effects
irregular schedules, shift work, and stress all contribute to widespread sleep deprivation.
The Consequences of Poor Sleep
Sleep deprivation isn't just about feeling tired - it has serious consequences:
- Cognitive Impairment: Affects concentration, judgment, and decision-making skills.
- Emotional Instability: More likely to experience mood disorders.
- Increased Risk of Accidents: Sleepy driving is as dangerous as drunk driving."""
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_path = f.name
    
    try:
        print(f"📄 Processing test document: {temp_path}")
        
        # Test document encoding with page fix
        vectorstore = encode_document(temp_path, chunk_size=200, chunk_overlap=50)
        print("✅ Document encoded successfully")
        
        # Test CRAG with improved scoring
        query = "What are two cognitive effects of sleep deprivation?"
        
        print(f"\n🔍 Testing query: {query}")
        
        # Initialize CRAG with file path (not vectorstore)
        crag = CRAG(
            file_path=temp_path,
            lower_threshold=0.3,
            upper_threshold=0.7
        )
        
        # Get response
        print("\n🤖 Running CRAG...")
        result = crag.run_with_sources(query)
        
        print(f"\n📊 CRAG Result:")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources count: {len(result.get('sources', []))}")
        
        # Check source chunks - handle both dict and tuple formats
        sources = result.get('sources', [])
        for i, source in enumerate(sources):
            print(f"\nSource {i+1}:")
            
            # Handle tuple format (text, metadata)
            if isinstance(source, tuple) and len(source) == 2:
                text, metadata = source
                score = metadata.get('score', 'N/A')
                page = metadata.get('page', 'N/A')
                print(f"  Score: {score}")
                print(f"  Page: {page}")
                print(f"  Text: {text[:100]}...")
                
                # Check if this is the problematic health benefits chunk
                if "health benefits" in text.lower() and "cognitive" not in text.lower():
                    print(f"  ⚠️  Found generic health benefits chunk - score should be low!")
                    if score != 'N/A' and score > 0.5:
                        print(f"  ❌ ERROR: Health benefits chunk has high score ({score})")
                    else:
                        print(f"  ✅ GOOD: Health benefits chunk has acceptable score ({score})")
                
                # Check if this answers the cognitive effects question
                if "cognitive" in text.lower() and ("concentration" in text.lower() or "judgment" in text.lower() or "decision-making" in text.lower()):
                    print(f"  ✅ RELEVANT: This chunk directly answers the cognitive effects question")
                    if score != 'N/A' and score < 0.6:
                        print(f"  ⚠️  WARNING: Relevant chunk has low score ({score})")
            
            # Handle dict format
            elif isinstance(source, dict):
                score = source.get('score', 'N/A')
                page = source.get('page', 'N/A')
                text = source.get('text', '')
                print(f"  Score: {score}")
                print(f"  Page: {page}")
                print(f"  Text: {text[:100]}...")
                
                # Same checks as above
                if "health benefits" in text.lower() and "cognitive" not in text.lower():
                    print(f"  ⚠️  Found generic health benefits chunk - score should be low!")
                    if score != 'N/A' and score > 0.5:
                        print(f"  ❌ ERROR: Health benefits chunk has high score ({score})")
                    else:
                        print(f"  ✅ GOOD: Health benefits chunk has acceptable score ({score})")
                
                if "cognitive" in text.lower() and ("concentration" in text.lower() or "judgment" in text.lower() or "decision-making" in text.lower()):
                    print(f"  ✅ RELEVANT: This chunk directly answers the cognitive effects question")
                    if score != 'N/A' and score < 0.6:
                        print(f"  ⚠️  WARNING: Relevant chunk has low score ({score})")
            else:
                print(f"  Format: {type(source)} - {str(source)[:100]}...")
        
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
    test_page_number_fix()
