#!/usr/bin/env python3
"""
Test script for colored source references in CRAG responses
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crag import CRAG

def test_colored_sources():
    """Test that source references are properly colored in responses"""
    print("🧪 TESTING COLORED SOURCE REFERENCES")
    print("=" * 60)
    
    # Test document path
    test_doc = "data/Understanding_Climate_Change (1).pdf"
    
    if not os.path.exists(test_doc):
        print("❌ Test document not found. Looking for available documents...")
        # Look for any PDF in the directory
        for file in os.listdir("."):
            if file.endswith(".pdf"):
                test_doc = file
                print(f"✅ Found alternative document: {test_doc}")
                break
        else:
            print("❌ No test documents found. Exiting.")
            return
    
    try:
        print(f"📄 Using document: {test_doc}")
        print("🔧 Initializing CRAG with colored sources...")
        
        # Initialize CRAG
        crag = CRAG(
            file_path=test_doc,
            web_search_enabled=False,  # Focus on document sources
            upper_threshold=0.6,
            lower_threshold=0.3
        )
        
        # Test query
        query = "What causes climate change?"
        
        print(f"\n❓ Query: {query}")
        print("\n🔄 Processing...")
        
        # Get response with colored sources
        response = crag.run(query)
        
        print("\n📝 RAW RESPONSE (with HTML formatting):")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        # Check if source formatting is present
        if "[Source:" in response and "class=\"source-reference\"" in response:
            print("\n✅ SUCCESS: Source references are properly formatted with color!")
            print("   📋 The sources will appear in blue with background highlighting")
            print("   📋 Format: <span class=\"source-reference\">[Source: filename, page X]</span>")
        elif "[Source:" in response:
            print("\n⚠️  PARTIAL: Source references found but may not be fully formatted")
            print("   📋 Check if the CSS class is being applied correctly")
        else:
            print("\n❌ ISSUE: No source references found in response")
            print("   📋 The LLM may not be following the source formatting instructions")
        
        print(f"\n🎯 WHAT TO EXPECT IN STREAMLIT:")
        print("   💙 Source references should appear in blue color")
        print("   📦 Background highlighting for better visibility") 
        print("   ✨ Hover effects for interactivity")
        print("   📱 Clean, professional appearance")
        
    except Exception as e:
        print(f"\n❌ Error during test: {str(e)}")
        return

if __name__ == "__main__":
    test_colored_sources()
