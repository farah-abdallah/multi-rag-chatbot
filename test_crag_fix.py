"""
Test CRAG functionality after fixing the missing method
"""

def test_crag_fixed():
    """Test that CRAG works after adding the missing method"""
    try:
        from crag import CRAG
        
        # Create a simple test document
        test_content = "This is a test document about climate change. Global warming is caused by greenhouse gases."
        test_file = "test_crag_doc.txt"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        print("🧪 Testing CRAG with fixed _call_llm_with_retry method...")
        
        # Initialize CRAG with web search disabled for faster testing
        crag = CRAG(
            file_path=test_file,
            web_search_enabled=False,
            model="gemini-1.5-flash",
            max_tokens=500
        )
        
        print("✅ CRAG initialized successfully")
        
        # Test the run method
        query = "What causes climate change?"
        print(f"🔍 Testing query: {query}")
        
        response = crag.run(query)
        print(f"✅ CRAG response received: {response[:100]}...")
        
        # Test the enhanced run_with_sources method
        print("🔍 Testing run_with_sources method...")
        result_data = crag.run_with_sources(query)
        
        print(f"✅ Enhanced response received")
        print(f"📄 Source chunks: {len(result_data['source_chunks'])}")
        print(f"📋 Sources: {len(result_data['sources'])}")
        
        # Clean up
        import os
        if os.path.exists(test_file):
            os.remove(test_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        # Clean up on error
        import os
        if os.path.exists("test_crag_doc.txt"):
            try:
                os.remove("test_crag_doc.txt")
            except:
                pass
        return False

def main():
    print("🔧 CRAG Fix Validation")
    print("=" * 40)
    
    success = test_crag_fixed()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 CRAG FIX SUCCESSFUL!")
        print("✅ _call_llm_with_retry method added")
        print("✅ CRAG.run() method working")
        print("✅ CRAG.run_with_sources() method working")
        print("\n🚀 Ready to use in chatbot_app.py!")
    else:
        print("❌ CRAG FIX FAILED")
        print("Please check the errors above.")

if __name__ == "__main__":
    main()
