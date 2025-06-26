#!/usr/bin/env python3

"""
Test script to verify that all vision processing has been successfully removed
"""

def test_vision_removal():
    print("🧪 Testing that all vision processing has been removed...")
    
    try:
        # Test all RAG technique imports
        print("📋 Testing RAG technique imports...")
        
        from adaptive_rag import AdaptiveRAG
        print("✅ AdaptiveRAG imported successfully")
          from crag import CRAG  
        print("✅ CRAG imported successfully")
        
        from document_augmentation import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")
        
        from explainable_retrieval import ExplainableRetrieval
        print("✅ ExplainableRetrieval imported successfully")
        
        from evaluation_framework import AutomatedEvaluator
        print("✅ AutomatedEvaluator imported successfully")
        
        # Test basic initialization
        print("\n🔧 Testing basic initialization...")
        
        texts = ['Climate change is caused by greenhouse gases.', 'Solar energy is renewable.']
        
        adaptive_rag = AdaptiveRAG(texts=texts)
        print("✅ AdaptiveRAG initialized successfully")
        
        # CRAG expects a file path, not texts array
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Climate change is caused by greenhouse gases.\nSolar energy is renewable.")
            temp_file = f.name
        
        try:
            crag = CRAG(file_path=temp_file)
            print("✅ CRAG initialized successfully")
        finally:
            os.unlink(temp_file)
        
        # ExplainableRetrieval uses texts parameter
        from explainable_retrieval import ExplainableRetrieval
        explainable_rag = ExplainableRetrieval(texts=texts)
        print("✅ ExplainableRetrieval initialized successfully")
        
        evaluator = AutomatedEvaluator()
        print("✅ AutomatedEvaluator initialized successfully")
        
        # Test that no vision-related attributes exist
        print("\n🔍 Testing that vision attributes are removed...")
        
        # Check SimpleGeminiLLM classes don't have vision attributes
        simple_llm = adaptive_rag.llm
        if hasattr(simple_llm, 'vision_model'):
            print("❌ AdaptiveRAG SimpleGeminiLLM still has vision_model attribute")
            return False
        if hasattr(simple_llm, 'vision_enabled'):
            print("❌ AdaptiveRAG SimpleGeminiLLM still has vision_enabled attribute")
            return False
        print("✅ AdaptiveRAG vision attributes successfully removed")
        
        crag_llm = crag.llm
        if hasattr(crag_llm, 'vision_model'):
            print("❌ CRAG SimpleGeminiLLM still has vision_model attribute")
            return False
        if hasattr(crag_llm, 'vision_enabled'):
            print("❌ CRAG SimpleGeminiLLM still has vision_enabled attribute")  
            return False
        print("✅ CRAG vision attributes successfully removed")
        
        # Check that analyze_image methods are removed
        if hasattr(simple_llm, 'analyze_image'):
            print("❌ AdaptiveRAG SimpleGeminiLLM still has analyze_image method")
            return False
        if hasattr(crag_llm, 'analyze_image'):
            print("❌ CRAG SimpleGeminiLLM still has analyze_image method")
            return False
        print("✅ analyze_image methods successfully removed")
        
        print("\n✅ All vision processing successfully removed!")
        print("🎉 Your RAG system is now vision-free and should work without PyMuPDF/Pillow dependencies")
        return True
        
    except Exception as e:
        print(f"❌ Error during vision removal test: {e}")
        return False

if __name__ == "__main__":
    test_vision_removal()
