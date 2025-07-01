"""
Test Document Viewer Integration with CRAG
This script tests the enhanced CRAG functionality with document viewer support
"""

import os
import sys
from crag import CRAG

def test_crag_with_sources():
    """Test CRAG with source tracking functionality"""
    
    # Use a sample document if available
    test_documents = [
        "data/Understanding_Climate_Change (1).pdf",
        "README.md",
        "SETUP_GUIDE.md"
    ]
    
    # Find an available test document
    test_file = None
    for doc in test_documents:
        if os.path.exists(doc):
            test_file = doc
            break
    
    if not test_file:
        print("❌ No test documents found. Please ensure you have sample documents.")
        return False
    
    print(f"🧪 Testing CRAG with document viewer using: {test_file}")
    
    try:
        # Initialize CRAG
        crag_system = CRAG(
            file_path=test_file,
            web_search_enabled=False  # Disable web search for testing
        )
        
        # Test standard run method
        print("\n1. Testing standard run method...")
        query = "What is the main topic of this document?"
        response = crag_system.run(query)
        print(f"✅ Standard response received: {response[:100]}...")
        
        # Test enhanced run_with_sources method
        print("\n2. Testing enhanced run_with_sources method...")
        if hasattr(crag_system, 'run_with_sources'):
            result_data = crag_system.run_with_sources(query)
            
            print(f"✅ Enhanced response received")
            print(f"📄 Source chunks found: {len(result_data['source_chunks'])}")
            print(f"📋 Sources found: {len(result_data['sources'])}")
            
            # Show source details
            for i, chunk in enumerate(result_data['source_chunks'][:2]):  # Show first 2 chunks
                print(f"\nChunk {i+1}:")
                print(f"  Source: {chunk.get('source', 'Unknown')}")
                print(f"  Score: {chunk.get('score', 'N/A')}")
                print(f"  Text preview: {chunk.get('text', '')[:150]}...")
            
            return True
        else:
            print("❌ run_with_sources method not found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_document_viewer_functions():
    """Test document viewer utility functions"""
    try:
        from document_viewer import highlight_text_in_document, show_embedded_document_viewer
        print("\n3. Testing document viewer imports...")
        print("✅ Document viewer functions imported successfully")
        return True
    except Exception as e:
        print(f"❌ Document viewer import failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting CRAG Document Viewer Integration Test")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_crag_with_sources()
    test2_passed = test_document_viewer_functions()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"   CRAG with sources: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Document viewer: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Document viewer integration is ready.")
        print("\nTo use the enhanced features:")
        print("1. Run: streamlit run chatbot_app.py")
        print("2. Upload a document and select 'CRAG' technique")
        print("3. Ask a question and look for document viewer links")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
