#!/usr/bin/env python3
"""
Test Script for Multi-RAG Chatbot Setup

This script verifies that all components are properly installed and configured.
"""

import sys
import os
from pathlib import Path

def test_python_version():
    """Test Python version requirement"""
    print("🐍 Testing Python version...")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def test_required_files():
    """Test that required files exist"""
    print("\n📁 Testing required files...")
    
    required_files = [
        "chatbot_app.py",
        "adaptive_rag.py", 
        "crag.py",
        "document_augmentation.py",
        "helper_functions.py",
        "requirements_chatbot.txt"
    ]
    
    missing_files = []
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ {file_name}")
            missing_files.append(file_name)
    
    return len(missing_files) == 0

def test_environment_file():
    """Test .env file configuration"""
    print("\n🔑 Testing environment configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("   ❌ .env file not found")
        print("   💡 Create .env file with: GOOGLE_API_KEY=your_key_here")
        return False
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
            
        if 'GOOGLE_API_KEY=' not in content:
            print("   ❌ GOOGLE_API_KEY not found in .env file")
            return False
        
        if 'your_google_api_key_here' in content or 'your_key_here' in content:
            print("   ⚠️ .env file contains placeholder - please set your actual API key")
            return False
        
        # Try to load with python-dotenv
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('GOOGLE_API_KEY')
            
            if api_key and len(api_key) > 10:
                print("   ✅ GOOGLE_API_KEY configured")
                return True
            else:
                print("   ❌ GOOGLE_API_KEY appears to be invalid")
                return False
                
        except ImportError:
            print("   ⚠️ python-dotenv not installed - will check later")
            return True
            
    except Exception as e:
        print(f"   ❌ Error reading .env file: {e}")
        return False

def test_package_imports():
    """Test that required packages can be imported"""
    print("\n📦 Testing package imports...")
    
    packages = [
        ("streamlit", "streamlit"),
        ("langchain", "langchain_core.prompts"),
        ("faiss", "faiss"),
        ("python-dotenv", "dotenv"),
        ("google-generativeai", "google.generativeai"),
        ("sentence-transformers", "sentence_transformers"),
        ("numpy", "numpy"),
        ("pandas", "pandas")
    ]
    
    failed_imports = []
    
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name}")
            failed_imports.append(package_name)
        except Exception as e:
            print(f"   ⚠️ {package_name} (warning: {e})")
    
    if failed_imports:
        print(f"\n   💡 Install missing packages with:")
        print(f"      pip install -r requirements_chatbot.txt")
        return False
    
    return True

def test_sample_data():
    """Test sample data availability"""
    print("\n📄 Testing sample data...")
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("   ⚠️ data/ directory not found")
        print("   💡 Create data/ directory and add sample files for testing")
        return False
    
    sample_files = list(data_dir.glob("*"))
    if not sample_files:
        print("   ⚠️ No sample files found in data/ directory")
        print("   💡 Add some sample documents (PDF, TXT, CSV, JSON) for testing")
        return False
    
    print(f"   ✅ Found {len(sample_files)} sample files:")
    for file_path in sample_files[:5]:  # Show first 5
        print(f"      📄 {file_path.name}")
    
    return True

def test_basic_functionality():
    """Test basic RAG functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test imports
        from adaptive_rag import AdaptiveRAG
        from helper_functions import encode_from_string
        print("   ✅ RAG modules import successfully")
        
        # Test basic text processing
        sample_text = "This is a test document for the RAG system."
        vectorstore = encode_from_string(sample_text)
        print("   ✅ Text encoding works")
        
        # Test basic retrieval
        docs = vectorstore.similarity_search("test document", k=1)
        if docs and len(docs) > 0:
            print("   ✅ Document retrieval works")
        else:
            print("   ❌ Document retrieval failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic functionality test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("=" * 60)
    print("🤖 MULTI-RAG CHATBOT SETUP TEST")
    print("=" * 60)
    
    tests = [
        ("Python Version", test_python_version),
        ("Required Files", test_required_files),
        ("Environment Config", test_environment_file),
        ("Package Imports", test_package_imports),
        ("Sample Data", test_sample_data),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Error in {test_name} test: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your Multi-RAG Chatbot is ready to use!")
        print("\nNext steps:")
        print("   1. Run: streamlit run chatbot_app.py")
        print("   2. Or: python launch_chatbot.py")
        print("   3. Or: python demo.py")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please fix the issues above before proceeding.")
        print("\nCommon solutions:")
        print("   • Install requirements: pip install -r requirements_chatbot.txt")
        print("   • Set up .env file with your Google API key")
        print("   • Ensure all project files are present")
    
    return passed == total

def main():
    """Main test function"""
    try:
        success = run_comprehensive_test()
        
        if success:
            # Offer to run demo
            print("\n" + "=" * 60)
            response = input("🚀 Would you like to run a quick demo? (y/n): ").lower().strip()
            
            if response == 'y':
                print("\n🔄 Starting demo...")
                try:
                    import demo
                    demo.main()
                except Exception as e:
                    print(f"Demo failed: {e}")
        
        return success
        
    except KeyboardInterrupt:
        print("\n\n👋 Test cancelled by user.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ Setup verification complete!")
    else:
        print("\n❌ Setup verification failed. Please review the errors above.")
        sys.exit(1)
