#!/usr/bin/env python3

"""
Test CRAG scoring for mental and emotional health queries
"""

import tempfile
import os
from crag import CRAG

def test_mental_health_scoring():
    # Create test document with both physical and mental health content
    test_content = '''The Health Benefits of Sleep

a) Physical Health
Getting enough sleep helps the body regulate vital systems:

• Immune system: Sleep supports immune function.
• Heart health: Helps maintain healthy blood pressure and reduces heart disease risk.
• Metabolism: Impacts how the body processes glucose, linked to weight gain and diabetes.

b) Mental and Emotional Health
Adequate sleep improves cognitive functions (attention, problem-solving, creativity), processes information and forms memories, and improves emotional regulation, reducing mood swings, anxiety, and depression risk.

c) Sleep Quality Factors
Temperature, noise levels, and light exposure all affect sleep quality.'''

    # Create a temporary file that persists
    temp_file = None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file = f.name
    
    # Ensure file is written and accessible
    import time
    time.sleep(0.1)

    try:
        print("🧪 Testing CRAG scoring for mental and emotional health query")
        print("=" * 70)
        
        # Verify the file was created and has content
        if not os.path.exists(temp_file):
            print(f"❌ ERROR: Temporary file {temp_file} does not exist")
            return False
            
        with open(temp_file, 'r') as f:
            file_content = f.read()
            print(f"✅ Temporary file created: {temp_file}")
            print(f"📁 File size: {len(file_content)} characters")
            print(f"📄 Content preview: {file_content[:100]}...")
        
        # Initialize CRAG with the test document
        print("\n🔧 Initializing CRAG...")
        crag = CRAG(file_path=temp_file, web_search_enabled=False)
        
        # Test query about mental and emotional health
        query = "How does sleep support mental and emotional health?"
        print(f"🔍 Query: {query}")
        print()
        
        # Run CRAG and see the scoring and chunk selection
        response = crag.run_with_sources(query)
        
        print("\n" + "=" * 70)
        print("🎯 CRAG Response:")
        print(response)
        
        print("\n" + "=" * 70)
        print("📊 Source chunks stored for highlighting:")
        if hasattr(crag, '_last_source_chunks'):
            for i, chunk in enumerate(crag._last_source_chunks, 1):
                print(f"\n  Chunk {i}:")
                print(f"    Score: {chunk.get('score', 'Unknown')}")
                print(f"    Text: '{chunk['text'][:150]}{'...' if len(chunk['text']) > 150 else ''}'")
        else:
            print("  No source chunks found")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        return False
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

if __name__ == "__main__":
    success = test_mental_health_scoring()
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
