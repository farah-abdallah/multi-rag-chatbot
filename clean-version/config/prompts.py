"""
System prompts and templates for the Multi-RAG Chatbot.
"""

# System prompts
SYSTEM_PROMPT = """You are a helpful AI assistant specialized in answering questions based on provided documents and web search results. 

Your capabilities include:
1. Analyzing and extracting information from multiple documents
2. Performing web searches when needed
3. Providing accurate, well-sourced answers
4. Highlighting relevant sources and citations

Guidelines:
- Always cite your sources when providing information
- If information is not available in the provided context, clearly state this
- Use web search to supplement document information when appropriate
- Provide clear, concise, and accurate responses
- Highlight the most relevant parts of sources when possible"""

CRAG_SYSTEM_PROMPT = """You are an expert information retrieval and analysis system. Your task is to:

1. Analyze the user's query and determine the best approach
2. Retrieve relevant information from available documents
3. Evaluate the quality and relevance of retrieved information
4. Perform web search if document information is insufficient
5. Synthesize a comprehensive answer from all available sources

Always prioritize accuracy and provide clear source attribution."""

EVALUATION_PROMPT = """Evaluate the relevance and quality of the following information for answering the user's question:

Question: {question}
Retrieved Information: {context}

Please assess:
1. Relevance to the question (0-10 scale)
2. Completeness of information (0-10 scale)
3. Reliability of sources (0-10 scale)
4. Whether additional information is needed

Provide your assessment and recommendation."""

KNOWLEDGE_REFINEMENT_PROMPT = """Based on the user's question and the available information, refine and enhance the knowledge base by:

1. Identifying key concepts and relationships
2. Highlighting the most relevant information
3. Organizing information in a logical structure
4. Suggesting additional search queries if needed

Question: {question}
Available Information: {context}

Provide a refined knowledge summary."""

WEB_SEARCH_PROMPT = """Generate effective search queries for the following question:

Question: {question}
Context: {context}

Provide 2-3 search queries that would help find additional relevant information."""

DOCUMENT_ANALYSIS_PROMPT = """Analyze the following document content and extract key information relevant to potential user questions:

Document: {document_content}

Provide:
1. Main topics and themes
2. Key facts and data points
3. Important relationships and connections
4. Potential question areas this document could answer"""

RESPONSE_SYNTHESIS_PROMPT = """Synthesize a comprehensive answer using the following sources:

Question: {question}
Document Sources: {document_context}
Web Search Results: {web_context}

Requirements:
1. Provide a clear, accurate answer
2. Cite all sources appropriately
3. Highlight the most relevant information
4. Note any limitations or uncertainties
5. Suggest follow-up questions if appropriate"""

# Template for source citation
SOURCE_CITATION_TEMPLATE = """
**Source**: {source_name}
**Type**: {source_type}
**Relevance**: {relevance_score}/10
**Key Points**: {key_points}
"""

# Template for web search results
WEB_RESULT_TEMPLATE = """
**Title**: {title}
**URL**: {url}
**Summary**: {summary}
**Date**: {date}
"""

# Template for document chunks
DOCUMENT_CHUNK_TEMPLATE = """
**Document**: {document_name}
**Page/Section**: {page_info}
**Content**: {content}
**Metadata**: {metadata}
"""
