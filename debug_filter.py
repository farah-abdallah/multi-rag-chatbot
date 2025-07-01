chunk_text = 'In this document, we will explore sleep effects.'
generic_phrases = [
    'in this document', 'we will explore', 'we will examine', 
    'this section discusses', 'the following', 'as mentioned',
    'it is important to note', 'furthermore', 'in addition'
]
print(f'Chunk text: "{chunk_text}"')
print(f'Length: {len(chunk_text)}')
print(f'Lower case: "{chunk_text.lower()}"')
has_generic = any(phrase in chunk_text.lower() for phrase in generic_phrases)
print(f'Has generic phrase: {has_generic}')
for phrase in generic_phrases:
    if phrase in chunk_text.lower():
        print(f'  Found phrase: "{phrase}"')
is_short = len(chunk_text) < 100
print(f'Is short (<100 chars): {is_short}')
should_skip = has_generic and is_short
print(f'Should skip: {should_skip}')
