#!/usr/bin/env python3
"""
Test script to verify that document highlighting uses a consistent color
"""

import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_viewer import highlight_text_in_document

def test_consistent_highlighting():
    """Test that multiple chunks get highlighted with the same color"""
    
    print("🎨 Testing Consistent Highlighting Colors")
    print("=" * 50)
    
    # Create a test document content
    test_content = """
    Mental Health and Sleep

    Sleep plays a crucial role in mental and emotional health. Good quality sleep helps regulate emotions and reduces stress.

    Physical Health Benefits

    Sleep is also important for physical recovery and immune system function.

    Cognitive Function

    Adequate sleep improves concentration, memory, and decision-making abilities.
    """
    
    # Write test content to a temporary file
    test_file = "test_highlight_colors.txt"
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Create test chunks that should be highlighted
        test_chunks = [
            {
                'text': 'Sleep plays a crucial role in mental and emotional health. Good quality sleep helps regulate emotions and reduces stress.',
                'score': 0.9
            },
            {
                'text': 'Adequate sleep improves concentration, memory, and decision-making abilities.',
                'score': 0.8
            },
            {
                'text': 'Sleep is also important for physical recovery and immune system function.',
                'score': 0.7
            }
        ]
        
        print(f"📄 Test document: {test_file}")
        print(f"📦 Test chunks: {len(test_chunks)}")
        
        # Apply highlighting
        highlighted_html = highlight_text_in_document(test_file, test_chunks)
        
        # Check for color consistency
        print("\n🔍 Analyzing highlight colors...")
        import re
        
        # Find all highlight color instances
        color_pattern = r'background-color:\s*hsl\([^)]+\)'
        colors_found = re.findall(color_pattern, highlighted_html)
        
        print(f"Colors found: {len(colors_found)}")
        for i, color in enumerate(colors_found):
            print(f"  {i+1}. {color}")
        
        # Check if all colors are the same
        unique_colors = set(colors_found)
        print(f"\nUnique colors: {len(unique_colors)}")
        
        if len(unique_colors) == 1:
            print("✅ SUCCESS: All highlights use the same color!")
            print(f"   Consistent color: {list(unique_colors)[0]}")
        elif len(unique_colors) == 0:
            print("⚠️  WARNING: No highlights found")
        else:
            print("❌ ISSUE: Multiple different colors found")
            for color in unique_colors:
                print(f"   - {color}")
        
        # Show a snippet of the highlighted content
        print(f"\n📝 Highlighted content preview:")
        preview = highlighted_html[:500] + "..." if len(highlighted_html) > 500 else highlighted_html
        print(preview)
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n🧹 Cleaned up test file: {test_file}")

if __name__ == "__main__":
    test_consistent_highlighting()
