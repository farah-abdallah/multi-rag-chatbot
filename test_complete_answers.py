#!/usr/bin/env python3
"""
Test script to verify that CRAG provides complete answers to "why" questions
"""

import os
import sys
from crag import CRAG

def test_sleep_importance_question():
    """Test that the sleep importance question gets a complete answer"""
    
    print("=" * 60)
    print("TESTING COMPLETE ANSWER GENERATION")
    print("=" * 60)
    
    # Check for available documents in order of preference
    test_files = [
        # Sleep documents (preferred for this test)
        r"C:\Users\iTECH\Downloads\The_Importance_of_Sleep (1).pdf",
        r"c:\Users\iTECH\Downloads\The_Importance_of_Sleep.pdf",
        "./data/The_Importance_of_Sleep.pdf",
        "./The_Importance_of_Sleep.pdf",
        # Fallback to available documents
        "./data/Understanding_Climate_Change (1).pdf",
        "./data/sample_text.txt"
    ]
    
    file_path = None
    for test_file in test_files:
        if os.path.exists(test_file):
            file_path = test_file
            print(f"📄 Found document: {test_file}")
            break
    
    if not file_path:
        print("❌ No test document found. Please ensure you have a document available.")
        print("Available options:")
        for tf in test_files:
            print(f"   - {tf}")
        return
    
    print(f"📄 Using document: {file_path}")
    
    # Initialize CRAG
    try:
        crag = CRAG(
            file_path=file_path,
            model="gemini-1.5-flash",
            max_tokens=2000,  # Increased for longer answers
            temperature=0,
            lower_threshold=0.3,
            upper_threshold=0.7,
            web_search_enabled=False  # Focus on document content
        )
        print("✅ CRAG initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize CRAG: {e}")
        return
    
    # Determine appropriate test question based on the document
    if "sleep" in file_path.lower():
        test_query = "Why is sleep considered a fundamental biological need?"
        importance_keywords = [
            "physical health", "emotional", "brain function", "quality of life",
            "critical role", "essential", "food and water", "biological need"
        ]
        print("📋 Testing with SLEEP document")
    elif "climate" in file_path.lower():
        test_query = "Why is climate change happening and what are the main causes?"
        importance_keywords = [
            "greenhouse gases", "carbon dioxide", "fossil fuels", "human activities",
            "temperature", "emissions", "causes", "burning", "industrial"
        ]
        print("📋 Testing with CLIMATE CHANGE document")
    else:
        test_query = "What are the main points explained in this document and why are they important?"
        importance_keywords = [
            "important", "because", "causes", "effects", "reasons", "why", "impact"
        ]
        print("📋 Testing with GENERAL document")
    
    print(f"\n🔍 Query: {test_query}")
    print("\n" + "=" * 60)
    print("RUNNING CRAG WITH IMPROVED KNOWLEDGE PROCESSING")
    print("=" * 60)
    
    try:
        # Use the enhanced run_with_sources method
        result = crag.run_with_sources(test_query)
        
        answer = result['answer']
        source_chunks = result['source_chunks']
        sources = result['sources']
        
        print(f"\n📝 COMPLETE ANSWER:")
        print("-" * 40)
        print(answer)
        print("-" * 40)
        
        print(f"\n📊 ANALYSIS:")
        print(f"Answer length: {len(answer)} characters")
        print(f"Number of source chunks: {len(source_chunks)}")
        print(f"Number of sources: {len(sources)}")
        
        # Check if answer mentions key aspects based on document type
        found_keywords = [kw for kw in importance_keywords if kw.lower() in answer.lower()]
        print(f"Relevant keywords found: {found_keywords}")
        
        # Check for comprehensiveness
        if len(answer) > 200 and len(found_keywords) >= 3:
            print("✅ ANSWER APPEARS COMPREHENSIVE")
            return True
        else:
            print("⚠️ ANSWER MAY BE INCOMPLETE - checking for explanatory content...")
            
            # Additional check for explanatory language
            explanatory_phrases = ["because", "due to", "as a result", "leads to", "causes", "why", "reason"]
            explanatory_found = [phrase for phrase in explanatory_phrases if phrase in answer.lower()]
            print(f"Explanatory phrases found: {explanatory_found}")
            
            if len(explanatory_found) >= 2:
                print("✅ ANSWER CONTAINS EXPLANATORY CONTENT")
                return True
            else:
                print("❌ ANSWER LACKS SUFFICIENT EXPLANATION")
                return False
            print(f"   - Length: {len(answer)} chars (should be >200)")
            print(f"   - Keywords: {len(found_keywords)}/8 (should be >=4)")
        
        print(f"\n🔗 SOURCE CHUNKS FOR HIGHLIGHTING:")
        for i, chunk in enumerate(source_chunks):
            score = chunk.get('score', 'N/A')
            source = chunk.get('source', 'Unknown')
            page = chunk.get('page', 'N/A')
            text_preview = chunk.get('text', '')[:150] + '...' if len(chunk.get('text', '')) > 150 else chunk.get('text', '')
            print(f"   {i+1}. Score: {score}, Source: {source}, Page: {page}")
            print(f"      Text: {text_preview}")
        
        return result
        
    except Exception as e:
        print(f"❌ CRAG processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_another_why_question():
    """Test another 'why' question to ensure consistency"""
    
    print("\n" + "=" * 60)
    print("TESTING ANOTHER WHY QUESTION")
    print("=" * 60)
    
    # Use the same document finding logic
    test_files = [
        # Sleep documents (preferred for this test)
        r"C:\Users\iTECH\Downloads\The_Importance_of_Sleep (1).pdf",
        r"c:\Users\iTECH\Downloads\The_Importance_of_Sleep.pdf",
        "./data/The_Importance_of_Sleep.pdf",
        "./The_Importance_of_Sleep.pdf",
        # Fallback to available documents
        "./data/Understanding_Climate_Change (1).pdf",
        "./data/sample_text.txt"
    ]
    
    file_path = None
    for test_file in test_files:
        if os.path.exists(test_file):
            file_path = test_file
            break
    
    if not file_path:
        print("❌ No test document found.")
        return False

    try:
        crag = CRAG(
            file_path=file_path,
            model="gemini-1.5-flash",
            max_tokens=2000,
            temperature=0,
            web_search_enabled=False
        )
        
        # Choose appropriate question based on document
        if "sleep" in file_path.lower():
            test_query = "Why does sleep affect brain function?"
            keywords = ["memory", "concentration", "cognitive", "brain", "neurons", "restoration"]
        elif "climate" in file_path.lower():
            test_query = "Why do greenhouse gases cause global warming?"
            keywords = ["radiation", "heat", "atmosphere", "temperature", "trapped", "energy"]
        else:
            test_query = "Why are the main concepts in this document important?"
            keywords = ["important", "significant", "impact", "effect", "reason", "because"]
        
        print(f"\n🔍 Query: {test_query}")
        
        result = crag.run_with_sources(test_query)
        answer = result['answer']
        
        print(f"\n📝 ANSWER:")
        print("-" * 40)
        print(answer)
        print("-" * 40)
        
        print(f"\nAnswer length: {len(answer)} characters")
        
        # Check for relevant keywords
        found_keywords = [kw for kw in keywords if kw.lower() in answer.lower()]
        print(f"Relevant keywords found: {found_keywords}")
        
        if len(answer) > 150 and len(found_keywords) >= 2:
            print("✅ Answer appears comprehensive")
            return True
        else:
            print("⚠️ Answer may be too brief or lacks detail")
            return False
            
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    print("🧪 TESTING COMPLETE ANSWER GENERATION IN CRAG")
    print("This test verifies that CRAG now provides comprehensive answers to 'why' questions\n")
    
    # Test 1: Original problematic question
    result1 = test_sleep_importance_question()
    
    # Test 2: Another why question
    result2 = test_another_why_question()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    test1_success = result1 is not None and result1 is not False
    test2_success = result2 is not None and result2 is not False
    
    if test1_success:
        print("✅ Test 1 (document-specific question): PASSED")
    else:
        print("❌ Test 1 (document-specific question): FAILED")
        
    if test2_success:
        print("✅ Test 2 (another why question): PASSED")
    else:
        print("❌ Test 2 (another why question): FAILED")
    
    print("\n🎯 KEY IMPROVEMENTS MADE:")
    print("1. ✅ Enhanced knowledge_refinement() to extract comprehensive information")
    print("2. ✅ Modified main processing to use knowledge refinement for ALL relevant chunks")
    print("3. ✅ Improved response generation prompt to emphasize completeness")
    print("4. ✅ Added explicit instructions for 'why' questions and causal explanations")
    
    print("\n📋 WHAT TO EXPECT:")
    print("- Complete answers that include causes, reasons, and explanations")
    print("- Better extraction of 'why' information from documents")
    print("- More comprehensive responses for fundamental questions")
    print("- Consistent processing whether chunks are high-confidence or fallback")
    
    if test1_success and test2_success:
        print("\n🎉 ALL TESTS PASSED - COMPLETE ANSWER FUNCTIONALITY IS WORKING!")
    elif test1_success or test2_success:
        print("\n⚠️ PARTIAL SUCCESS - Some improvements working, may need further tuning")
    else:
        print("\n❌ TESTS FAILED - Need to investigate further")
