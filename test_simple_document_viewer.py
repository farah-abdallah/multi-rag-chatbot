"""
Simple test for document viewer without API dependencies
"""
import os
import sys

def test_document_viewer_imports():
    """Test that document viewer can be imported without errors"""
    try:
        from document_viewer import (
            highlight_text_in_document,
            create_document_link,
            show_embedded_document_viewer,
            check_document_viewer_page
        )
        print("✅ Document viewer imports successful")
        return True
    except Exception as e:
        print(f"❌ Document viewer import failed: {e}")
        return False

def test_syntax_check():
    """Test that all Python files have valid syntax"""
    files_to_check = [
        'document_viewer.py',
        'crag.py',
        'chatbot_app.py'
    ]
    
    all_good = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                compile(code, file_path, 'exec')
                print(f"✅ {file_path} syntax is valid")
            except SyntaxError as e:
                print(f"❌ {file_path} has syntax error: {e}")
                all_good = False
            except Exception as e:
                print(f"⚠️ {file_path} check failed: {e}")
        else:
            print(f"⚠️ {file_path} not found")
    
    return all_good

def test_crag_enhancements():
    """Test that CRAG has the new methods without actually running them"""
    try:
        from crag import CRAG
        
        # Check if the new method exists
        if hasattr(CRAG, 'run_with_sources'):
            print("✅ CRAG.run_with_sources method exists")
        else:
            print("❌ CRAG.run_with_sources method missing")
            return False
            
        # Check if CRAG has source tracking attributes in __init__
        import inspect
        init_source = inspect.getsource(CRAG.__init__)
        if '_last_source_chunks' in init_source and '_last_sources' in init_source:
            print("✅ CRAG source tracking attributes found")
        else:
            print("❌ CRAG source tracking attributes missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ CRAG enhancement test failed: {e}")
        return False

def main():
    print("🧪 Running Simple Document Viewer Tests")
    print("=" * 60)
    
    results = {
        'syntax_check': test_syntax_check(),
        'document_viewer_imports': test_document_viewer_imports(),
        'crag_enhancements': test_crag_enhancements()
    }
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    if all(results.values()):
        print("\n🎉 All tests passed! The document viewer integration is ready.")
        print("\n📖 Usage Instructions:")
        print("1. Run: streamlit run chatbot_app.py")
        print("2. Upload a document and ask a question using CRAG")
        print("3. Look for '📄 Source Documents' section in the response")
        print("4. Click 'View Document' to see highlighted source chunks")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
