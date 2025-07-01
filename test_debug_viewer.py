import tempfile
import os
from document_viewer import highlight_text_in_document

# Create test content
test_content = '''Introduction to Sleep

In this document, we will explore sleep effects. Sleep deprivation has serious consequences:

• Cognitive Impairment: Affects concentration, judgment, and decision-making skills.
• Emotional Instability: More likely to experience mood disorders.'''

with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write(test_content)
    temp_file = f.name

try:
    # Test chunks with different relevance scores
    chunks_to_highlight = [
        {'text': 'In this document, we will explore sleep effects.', 'score': 0.3},  # Should be filtered out
        {'text': 'Cognitive Impairment: Affects concentration, judgment, and decision-making skills.', 'score': 0.8},  # Should be highlighted
        {'text': 'Short', 'score': 0.9},  # Should be filtered out (too short)        
    ]

    print('Testing document viewer filtering...')
    highlighted = highlight_text_in_document(temp_file, chunks_to_highlight)

    # Check results
    has_intro_highlighted = 'In this document' in highlighted and '<mark' in highlighted[highlighted.find('In this document'):highlighted.find('In this document') + 100]
    has_cognitive_highlighted = 'Cognitive Impairment' in highlighted and '<mark' in highlighted[highlighted.find('Cognitive Impairment'):highlighted.find('Cognitive Impairment') + 100]

    print(f'Intro text highlighted: {has_intro_highlighted}')
    print(f'Cognitive text highlighted: {has_cognitive_highlighted}')

    if has_intro_highlighted:
        print('❌ FAILED: Low-relevance intro text was highlighted')
    elif has_cognitive_highlighted:
        print('✅ PASSED: High-relevance chunk was highlighted, low-relevance intro was filtered out')
    else:
        print('⚠️  WARNING: No chunks were highlighted')

finally:
    os.unlink(temp_file)
