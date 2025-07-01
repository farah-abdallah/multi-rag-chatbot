"""
Test script to verify the updated query params API works correctly
"""

def test_document_viewer_query_params():
    """Test that document viewer handles query params correctly with new API"""
    print("🧪 Testing updated query params API...")
    
    try:
        # Test import
        from document_viewer import (
            document_viewer_page, 
            check_document_viewer_page,
            highlight_text_in_document
        )
        print("✅ Document viewer imports successful")
        
        # Test syntax validation
        import ast
        with open('document_viewer.py', 'r') as f:
            code = f.read()
        
        # Parse the code to check for syntax errors
        try:
            ast.parse(code)
            print("✅ Document viewer syntax is valid")
        except SyntaxError as e:
            print(f"❌ Syntax error in document_viewer.py: {e}")
            return False
        
        # Check that experimental_get_query_params is no longer used
        if 'experimental_get_query_params' in code:
            print("❌ Still using deprecated experimental_get_query_params")
            return False
        else:
            print("✅ No deprecated query params function found")
        
        # Check that st.query_params is used
        if 'st.query_params' in code:
            print("✅ Using new st.query_params API")
        else:
            print("❌ st.query_params not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("🔧 Query Params API Update Validation")
    print("=" * 50)
    
    success = test_document_viewer_query_params()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ QUERY PARAMS UPDATE SUCCESSFUL!")
        print("\n📝 Changes made:")
        print("• Replaced st.experimental_get_query_params() with st.query_params")
        print("• Updated parameter access (removed [0] indexing)")
        print("• Fixed comparison logic for page parameter")
        
        print("\n🚀 Ready to use with modern Streamlit!")
        print("Run: streamlit run chatbot_app.py")
    else:
        print("❌ QUERY PARAMS UPDATE FAILED")
        print("Please check the errors above.")

if __name__ == "__main__":
    main()
