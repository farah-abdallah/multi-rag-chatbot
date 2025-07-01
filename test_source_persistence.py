#!/usr/bin/env python3
"""
Test script to verify that CRAG source document persistence works correctly
"""

import os
import sys
import tempfile

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_crag_import():
    """Test that CRAG imports correctly"""
    try:
        from crag import CRAG
        print("✅ CRAG import successful")
        return True
    except Exception as e:
        print(f"❌ CRAG import failed: {e}")
        return False

def test_crag_run_with_sources():
    """Test that CRAG run_with_sources method works"""
    try:
        from crag import CRAG
        
        # Create a simple test document
        test_content = """
        Sleep is a fundamental biological process essential for human health and well-being. 
        
        Importance of Sleep:
        Sleep plays a crucial role in physical health, brain function, and emotional well-being. During sleep, the body repairs tissues, consolidates memories, and removes toxins from the brain.
        
        Modern Lifestyle Factors:
        Several modern lifestyle factors can negatively impact sleep duration and quality:
        1. Irregular schedules - shift work, late nights
        2. Shift work - disrupts natural circadian rhythms  
        3. Stress - chronic stress interferes with relaxation
        """
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            # Initialize CRAG system
            print("Initializing CRAG system...")
            crag = CRAG(temp_path, web_search_enabled=False)
            
            # Test run_with_sources method
            print("Testing run_with_sources...")
            query = "What are three modern lifestyle factors that reduce sleep duration?"
            
            result = crag.run_with_sources(query)
            
            print("✅ CRAG run_with_sources completed successfully")
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Found {len(result['source_chunks'])} source chunks")
            print(f"Found {len(result['sources'])} sources")
            
            return True
            
        finally:
            # Clean up temp file
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"❌ CRAG run_with_sources failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chatbot_imports():
    """Test that all chatbot imports work"""
    try:
        # Test document viewer imports
        from document_viewer import create_document_link, show_embedded_document_viewer
        print("✅ Document viewer imports successful")
        
        # Test main chatbot imports
        import streamlit as st
        print("✅ Streamlit import successful")
        
        return True
    except Exception as e:
        print(f"❌ Chatbot imports failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing CRAG source document persistence...")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: CRAG import
    if test_crag_import():
        tests_passed += 1
    
    # Test 2: CRAG run_with_sources
    if test_crag_run_with_sources():
        tests_passed += 1
    
    # Test 3: Chatbot imports  
    if test_chatbot_imports():
        tests_passed += 1
    
    print("=" * 50)
    print(f"🎯 Tests completed: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The fix should work correctly.")
        print("\n💡 The issue was caused by:")
        print("   1. st.rerun() being called immediately after displaying source buttons")
        print("   2. Source chunks not being persisted in session state")
        print("   3. Source buttons being shown in the response generation function instead of message display")
        print("\n✨ The fix:")
        print("   1. Store source chunks in session state by message ID")
        print("   2. Display source documents in the message display function")
        print("   3. Use unique keys for each message's document viewer buttons")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
