#!/usr/bin/env python3
"""
Test script to verify clean source name display
"""

import os
import sys
from crag import CRAG

def test_source_name_cleaning():
    """Test that source names are displayed cleanly without full temp paths"""
    print("🧪 TESTING SOURCE NAME CLEANING")
    print("=" * 60)
    
    # Test the _create_source_name method directly
    from crag import CRAG
    crag = CRAG.__new__(CRAG)  # Create instance without initialization
    
    # Test various metadata scenarios
    test_cases = [
        {
            'name': 'Temp file path',
            'metadata': {
                'source': r'C:\Users\iTECH\AppData\Local\Temp\tmp4z39ztp1\The_Importance_of_Sleep (1).pdf',
                'page': 1,
                'paragraph': 2
            },
            'expected': 'The_Importance_of_Sleep (1).pdf, page 1, paragraph 2'
        },
        {
            'name': 'Regular file path',
            'metadata': {
                'source': r'documents\Sleep_Research.pdf',
                'page': 3
            },
            'expected': 'Sleep_Research.pdf, page 3'
        },
        {
            'name': 'Just filename',
            'metadata': {
                'source': 'Climate_Change.pdf',
                'paragraph': 5
            },
            'expected': 'Climate_Change.pdf, paragraph 5'
        },
        {
            'name': 'No page/paragraph',
            'metadata': {
                'source': r'/home/user/docs/Research.pdf'
            },
            'expected': 'Research.pdf'
        }
    ]
    
    print("Testing source name cleaning:")
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        result = crag._create_source_name(test_case['metadata'])
        expected = test_case['expected']
        passed = result == expected
        
        print(f"\n{i}. {test_case['name']}:")
        print(f"   Input: {test_case['metadata']}")
        print(f"   Result: '{result}'")
        print(f"   Expected: '{expected}'")
        print(f"   Status: {'✅ PASS' if passed else '❌ FAIL'}")
        
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL SOURCE NAME TESTS PASSED!")
        print("📋 Source names will now display cleanly as:")
        print("   • 'Document.pdf, page X, paragraph Y'")
        print("   • Instead of full temp paths")
    else:
        print("❌ SOME TESTS FAILED - needs debugging")
    
    return all_passed

if __name__ == "__main__":
    test_source_name_cleaning()
