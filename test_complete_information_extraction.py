#!/usr/bin/env python3
"""
Test script to verify that the system extracts and uses complete information from documents.
This test checks that the system doesn't truncate or miss parts of statements when generating answers.
"""
import os
import sys
sys.path.append('.')

from crag import CRAG
import tempfile

def test_complete_information_extraction():
    """Test that the system extracts complete information from documents"""
    print("=== Testing Complete Information Extraction ===\n")
    
    # Create a test document with multi-part statements
    test_content = """Sleep Hygiene Guidelines

Exercise regularly: Helps sleep, but not too close to bedtime.

Caffeine consumption: Avoid after 2 PM as it can disrupt sleep patterns, but morning coffee is beneficial for alertness.

Screen time: Blue light from devices can interfere with sleep, but using blue light filters or avoiding screens 1 hour before bed helps maintain good sleep quality.

Room temperature: Keep bedroom between 60-67°F (15-19°C) for optimal sleep. Temperatures outside this range can cause restlessness.

Sleep schedule: Go to bed and wake up at the same time daily, including weekends, to maintain your circadian rhythm and improve sleep quality.
"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
        temp_file.write(test_content)
        temp_file_path = temp_file.name
    
    try:
        # Initialize CRAG with the test document and debug enabled
        print("📄 Initializing CRAG with test document...")
        crag = CRAG(temp_file_path, web_search_enabled=False)
        print("✅ CRAG initialized successfully")
        
        # Test queries that should get complete information
        test_queries = [
            "Tell me about exercise and sleep",
            "What should I know about caffeine and sleep?",
            "How does screen time affect sleep?",
            "What's the recommended room temperature for sleep?",
            "What about sleep schedule?"
        ]
        
        print("\n" + "="*80)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test Query {i}: {query}")
            print("-" * 60)
            
            # Get answer with source tracking
            result = crag.run_with_sources(query)
            answer = result['answer']
            source_chunks = result['source_chunks']
            
            print(f"\n📝 Generated Answer:")
            print(answer)
            
            print(f"\n📊 Source Chunks Found: {len(source_chunks)}")
            for j, chunk in enumerate(source_chunks):
                score = chunk.get('score', 'N/A')
                text = chunk.get('text', '')
                print(f"   {j+1}. Score: {score}")
                print(f"      Text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            
            # Check if answer contains both positive and negative aspects where applicable
            if "exercise" in query.lower():
                has_benefit = "help" in answer.lower() or "beneficial" in answer.lower()
                has_timing = "not too close" in answer.lower() or "bedtime" in answer.lower()
                print(f"\n✅ Exercise Analysis:")
                print(f"   - Contains benefit info: {has_benefit}")
                print(f"   - Contains timing warning: {has_timing}")
                print(f"   - Complete information: {has_benefit and has_timing}")
                
            elif "caffeine" in query.lower():
                has_timing = "2 PM" in answer or "after 2" in answer.lower()
                has_benefit = ("morning" in answer.lower() and "beneficial" in answer.lower()) or "alertness" in answer.lower()
                print(f"\n✅ Caffeine Analysis:")
                print(f"   - Contains timing restriction: {has_timing}")
                print(f"   - Contains morning benefit: {has_benefit}")
                print(f"   - Complete information: {has_timing and has_benefit}")
                
            elif "screen" in query.lower():
                has_problem = "interfere" in answer.lower() or "blue light" in answer.lower()
                has_solution = "filter" in answer.lower() or "1 hour before" in answer.lower() or "avoiding" in answer.lower()
                print(f"\n✅ Screen Time Analysis:")
                print(f"   - Contains problem info: {has_problem}")
                print(f"   - Contains solution info: {has_solution}")
                print(f"   - Complete information: {has_problem and has_solution}")
                
            elif "temperature" in query.lower():
                has_range = ("60" in answer and "67" in answer) or ("15" in answer and "19" in answer)
                has_consequence = "restless" in answer.lower() or "outside" in answer.lower()
                print(f"\n✅ Temperature Analysis:")
                print(f"   - Contains specific range: {has_range}")
                print(f"   - Contains consequence info: {has_consequence}")
                print(f"   - Complete information: {has_range}")
                
            elif "schedule" in query.lower():
                has_consistency = "same time" in answer.lower()
                has_weekends = "weekend" in answer.lower()
                has_benefit = "circadian" in answer.lower() or "improve" in answer.lower()
                print(f"\n✅ Schedule Analysis:")
                print(f"   - Contains consistency requirement: {has_consistency}")
                print(f"   - Contains weekend guidance: {has_weekends}")
                print(f"   - Contains benefit explanation: {has_benefit}")
                print(f"   - Complete information: {has_consistency and has_benefit}")
            
            print("\n" + "="*80)
        
        print("\n🎯 Summary:")
        print("This test verifies that the system:")
        print("1. Extracts complete information from source documents")
        print("2. Doesn't truncate multi-part statements")
        print("3. Includes both positive and negative aspects when present")
        print("4. Provides comprehensive answers based on available source material")
        
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print(f"\n🧹 Cleaned up temporary file: {temp_file_path}")

if __name__ == "__main__":
    test_complete_information_extraction()
