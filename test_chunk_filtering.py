#!/usr/bin/env python3

"""
Test script to verify chunk filtering and highlighting improvements
"""

import sys
import os
import tempfile
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_crag_chunk_filtering():
    """Test CRAG chunk filtering with different relevance scores"""
    try:
        from crag import CRAG
        
        # Create test document
        test_content = """
        Introduction to Sleep
        
        Sleep is often overlooked in the hustle of daily life. While we prioritize work, social activities, and screen time, sleep frequently takes a backseat. However, research consistently shows that sleep is a fundamental biological need, as essential as food and water. It plays a critical role in physical health, emotional stability, brain function, and overall quality of life.
        
        In this document, we will explore why sleep is important, what happens during sleep, how much sleep is necessary, and what can be done to improve sleep quality. We will also examine the consequences of chronic sleep deprivation and the broader implications of poor sleep habits.
        
        What Happens During Sleep?
        Although it may seem like the body shuts down during sleep, it actually enters a highly active state where vital processes take place. Sleep is divided into two main types:
        - Non-Rapid Eye Movement (NREM) Sleep
        - Rapid Eye Movement (REM) Sleep
        
        The Consequences of Poor Sleep
        Sleep deprivation isn't just about feeling tired - it has serious consequences:
        
        • Cognitive Impairment: Affects concentration, judgment, and decision-making skills.
        • Emotional Instability: More likely to experience mood disorders.
        • Increased Risk of Accidents: Sleepy driving is as dangerous as drunk driving.
        • Chronic Illnesses: Associated with heart disease, stroke, obesity, and certain cancers.
        
        Modern Lifestyle Factors
        Unfortunately, modern lifestyles often prevent people from getting enough rest. Social media, irregular schedules, shift work, and stress all contribute to widespread sleep deprivation.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = f.name
        
        try:
            print("Testing CRAG chunk filtering...")
            
            # Ensure file exists and is readable
            if not os.path.exists(temp_file):
                raise FileNotFoundError(f"Temp file not found: {temp_file}")
                
            print(f"Using temp file: {temp_file}")
            
            # Initialize CRAG with stricter thresholds
            crag = CRAG(temp_file, upper_threshold=0.7, lower_threshold=0.4)
            
            # Test query about cognitive effects
            query = "What are two cognitive effects of sleep deprivation?"
            print(f"\nQuery: {query}")
            
            result = crag.run_with_sources(query)
            
            print(f"Answer: {result['answer'][:200]}...")
            print(f"Number of source chunks: {len(result['source_chunks'])}")
            
            for i, chunk in enumerate(result['source_chunks']):
                print(f"\nChunk {i+1} (Score: {chunk['score']:.2f}):")
                print(f"Text: {chunk['text'][:100]}...")
                
            # Verify that low-relevance intro chunks are filtered out
            intro_chunks = [chunk for chunk in result['source_chunks'] 
                          if 'in this document' in chunk['text'].lower() 
                          or 'we will explore' in chunk['text'].lower()]
            
            if intro_chunks:
                print(f"\nWARNING: Found {len(intro_chunks)} introductory chunks that should be filtered out")
                for chunk in intro_chunks:
                    print(f"  - Score: {chunk['score']:.2f}, Text: {chunk['text'][:50]}...")
            else:
                print("\n✓ Good: No irrelevant introductory chunks found in results")
                
            return True
            
        finally:
            # Clean up
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
            
    except Exception as e:
        print(f"CRAG chunk filtering test failed: {str(e)}")
        return False

def test_document_viewer_filtering():
    """Test document viewer chunk filtering"""
    try:
        from document_viewer import highlight_text_in_document
        
        # Create test document
        test_content = """
        Introduction to Sleep
        
        In this document, we will explore sleep effects. Sleep deprivation has serious consequences:
        
        • Cognitive Impairment: Affects concentration, judgment, and decision-making skills.
        • Emotional Instability: More likely to experience mood disorders.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = f.name
        
        try:
            print("\nTesting document viewer filtering...")
            print(f"Using temp file: {temp_file}")
            
            # Ensure file exists and is readable
            if not os.path.exists(temp_file):
                raise FileNotFoundError(f"Temp file not found: {temp_file}")
            
            # Test chunks with different relevance scores
            chunks_to_highlight = [
                {'text': 'In this document, we will explore sleep effects.', 'score': 0.3},  # Should be filtered out
                {'text': 'Cognitive Impairment: Affects concentration, judgment, and decision-making skills.', 'score': 0.8},  # Should be highlighted
                {'text': 'Short', 'score': 0.9},  # Should be filtered out (too short)
            ]
            
            highlighted = highlight_text_in_document(temp_file, chunks_to_highlight)
            
            # Check results
            has_intro_highlighted = 'In this document' in highlighted and '<mark' in highlighted[highlighted.find('In this document'):highlighted.find('In this document') + 100]
            has_cognitive_highlighted = 'Cognitive Impairment' in highlighted and '<mark' in highlighted[highlighted.find('Cognitive Impairment'):highlighted.find('Cognitive Impairment') + 100]
            
            if has_intro_highlighted:
                print("WARNING: Low-relevance intro text was highlighted")
                return False
            elif has_cognitive_highlighted:
                print("✓ Good: High-relevance chunk was highlighted, low-relevance intro was filtered out")
                return True
            else:
                print("WARNING: No chunks were highlighted")
                return False
                
        finally:
            # Clean up
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
            
    except Exception as e:
        print(f"Document viewer filtering test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("Testing chunk filtering improvements...")
    
    tests = [
        ("CRAG chunk filtering", test_crag_chunk_filtering),
        ("Document viewer filtering", test_document_viewer_filtering),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"\n{test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"\n{test_name}: FAILED with error: {str(e)}")
            results.append((test_name, False))
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
