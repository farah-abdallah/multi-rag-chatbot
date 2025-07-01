"""
Safe test for document viewer functionality that avoids encoding issues
"""
import os

def test_document_viewer_safe():
    """Test document viewer with a simple text file to avoid encoding issues"""
    print("🧪 Testing document viewer functionality...")
    
    try:
        from document_viewer import highlight_text_in_document
        
        # Create a simple test file to avoid encoding issues
        test_file_path = "test_safe_document.txt"
        test_content = "This is a test document.\nIt contains some sample text for testing.\nThe document viewer should highlight specific chunks."
        
        # Write test file with explicit UTF-8 encoding
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Test highlighting
        chunks_to_highlight = [
            {'text': 'test document', 'score': 0.9},
            {'text': 'sample text', 'score': 0.8}
        ]
        
        highlighted_content = highlight_text_in_document(test_file_path, chunks_to_highlight)
        
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        
        # Check if highlighting worked
        if '<mark' in highlighted_content and 'test document' in highlighted_content:
            print("✅ Document highlighting functionality works")
            return True
        else:
            print("❌ Document highlighting failed")
            print(f"Content preview: {highlighted_content[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Document viewer functionality test failed: {e}")
        # Clean up test file if it exists
        if os.path.exists("test_safe_document.txt"):
            try:
                os.remove("test_safe_document.txt")
            except:
                pass
        return False

def test_query_params_update():
    """Test that query params are updated correctly"""
    print("🧪 Testing query params update...")
    
    try:
        # Read the document_viewer.py file
        with open('document_viewer.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for deprecated function
        if 'experimental_get_query_params' in content:
            print("❌ Still using deprecated experimental_get_query_params")
            return False
        
        # Check for new function
        if 'st.query_params' in content:
            print("✅ Using new st.query_params API")
        else:
            print("❌ st.query_params not found")
            return False
        
        # Check syntax
        import ast
        try:
            ast.parse(content)
            print("✅ Document viewer syntax is valid")
        except SyntaxError as e:
            print(f"❌ Syntax error: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Query params test failed: {e}")
        return False

def main():
    print("🛡️ Safe Document Viewer Test")
    print("=" * 50)
    
    test1 = test_query_params_update()
    test2 = test_document_viewer_safe()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Query params update: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"   Document viewer safe test: {'✅ PASSED' if test2 else '❌ FAILED'}")
    
    if test1 and test2:
        print("\n🎉 All tests passed!")
        print("✅ Query params updated successfully")
        print("✅ Document viewer functionality working")
        print("\n🚀 Ready to use: streamlit run chatbot_app.py")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
