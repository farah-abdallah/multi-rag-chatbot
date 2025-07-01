#!/usr/bin/env python3
"""
Test script to debug PDF page metadata assignment
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def test_pdf_page_metadata():
    # Load test document
    pdf_path = r"c:\Users\iTECH\work\multi-rag-chatbot-5\data\Understanding_Climate_Change (1).pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"=== Original PDF documents (before chunking) ===")
    for i, doc in enumerate(documents):
        print(f"Document {i}:")
        print(f"  Page: {doc.metadata.get('page', 'MISSING')}")
        print(f"  Source: {doc.metadata.get('source', 'MISSING')}")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    print(f"=== Chunks after splitting ===")
    for i, doc in enumerate(texts):
        print(f"Chunk {i}:")
        print(f"  Page: {doc.metadata.get('page', 'MISSING')}")
        print(f"  Source: {doc.metadata.get('source', 'MISSING')}")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print()

if __name__ == "__main__":
    test_pdf_page_metadata()
