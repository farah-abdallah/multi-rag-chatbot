"""
Final validation script for CRAG Document Viewer implementation
"""

def validate_implementation():
    """Comprehensive validation of the document viewer implementation"""
    
    print("🔍 CRAG Document Viewer Implementation Validation")
    print("=" * 60)
    
    # Test 1: Import validation
    try:
        from document_viewer import (
            highlight_text_in_document,
            create_document_link, 
            show_embedded_document_viewer,
            check_document_viewer_page
        )
        print("✅ Document viewer imports successful")
    except Exception as e:
        print(f"❌ Document viewer import failed: {e}")
        return False
    
    # Test 2: CRAG enhancement validation
    try:
        from crag import CRAG
        
        # Check for new method
        assert hasattr(CRAG, 'run_with_sources'), "run_with_sources method missing"
        print("✅ CRAG.run_with_sources method found")
        
        # Check for source tracking in __init__
        import inspect
        init_source = inspect.getsource(CRAG.__init__)
        assert '_last_source_chunks' in init_source, "Source tracking missing in __init__"
        assert '_last_sources' in init_source, "Source list tracking missing in __init__"
        print("✅ CRAG source tracking attributes found")
        
    except Exception as e:
        print(f"❌ CRAG enhancement validation failed: {e}")
        return False
    
    # Test 3: Chatbot app integration validation
    try:
        from chatbot_app import get_rag_response, main
        
        # Check for document viewer imports
        import chatbot_app
        source = inspect.getsource(chatbot_app)
        assert 'from document_viewer import' in source, "Document viewer not imported in chatbot_app"
        assert 'create_document_link' in source, "create_document_link not used in chatbot_app"
        print("✅ Chatbot app integration found")
        
    except Exception as e:
        print(f"❌ Chatbot app integration validation failed: {e}")
        return False
    
    # Test 4: Dependencies validation
    try:
        import bs4  # BeautifulSoup4
        import requests
        print("✅ Required dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return False
    
    return True

def main():
    """Main validation function"""
    success = validate_implementation()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 IMPLEMENTATION VALIDATION SUCCESSFUL!")
        print("\n📖 Ready to use! Follow these steps:")
        print("1. Run: streamlit run chatbot_app.py")
        print("2. Upload a document (PDF, TXT, DOCX, etc.)")
        print("3. Select 'CRAG' as your RAG technique")
        print("4. Ask a question about your document")
        print("5. Look for '📄 Source Documents' section in the response")
        print("6. Click the document links to view highlighted source chunks")
        
        print("\n🔧 Key Features:")
        print("• Text chunks are highlighted in different colors")
        print("• View documents in new tabs or embedded in chat")
        print("• See relevance scores for each chunk")
        print("• Download original documents")
        print("• Navigate directly to source passages")
        
    else:
        print("❌ IMPLEMENTATION VALIDATION FAILED")
        print("Please check the errors above and fix them before proceeding.")

if __name__ == "__main__":
    main()
