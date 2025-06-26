"""
Debug script to trace context and faithfulness evaluation
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation_framework import EvaluationManager, AutomatedEvaluator
import sqlite3

def inspect_evaluation_call():
    """Trace what happens during evaluation with debug prints"""
    
    # Create evaluator instance
    evaluator = AutomatedEvaluator()
    
    # Test cases
    test_cases = [
        {
            "name": "Empty context",
            "query": "What is climate change?",
            "response": "Climate change refers to long-term shifts in global temperatures and weather patterns.",
            "context": ""
        },
        {
            "name": "Valid context",
            "query": "What is climate change?", 
            "response": "Climate change refers to long-term shifts in global temperatures and weather patterns.",
            "context": "Climate change is a long-term change in the average weather patterns that have come to define Earth's local, regional and global climates. These changes have a broad range of observed effects that are synonymous with the term."
        },
        {
            "name": "Placeholder context",
            "query": "What is climate change?",
            "response": "Climate change refers to long-term shifts in global temperatures and weather patterns.",
            "context": "Context from uploaded documents"
        }
    ]
    
    print("=== DEBUGGING FAITHFULNESS EVALUATION ===\n")
    
    for test_case in test_cases:
        print(f"Test Case: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        print(f"Response: {test_case['response'][:50]}...")
        print(f"Context: '{test_case['context']}'")
        print(f"Context length: {len(test_case['context'])}")
        
        # Call faithfulness evaluation directly
        faithfulness_score = evaluator.evaluate_faithfulness_heuristic(
            test_case['context'], 
            test_case['response']
        )
        
        print(f"Faithfulness Score: {faithfulness_score}")
        print("-" * 50)
    
    # Now test with full evaluation manager
    print("\n=== TESTING FULL EVALUATION MANAGER ===\n")
    
    eval_manager = EvaluationManager("debug_evaluation.db")
    
    for test_case in test_cases:
        print(f"Full evaluation for: {test_case['name']}")
        
        query_id = eval_manager.evaluate_rag_response(
            query=test_case['query'],
            response=test_case['response'],
            technique="Test Technique",
            document_sources=["test.txt"],
            context=test_case['context'],
            processing_time=1.0,
            session_id="debug_session"
        )
          # Check what was stored
        conn = sqlite3.connect("debug_evaluation.db")
        cursor = conn.cursor()
        cursor.execute("SELECT faithfulness_score, query FROM evaluations WHERE query_id = ?", (query_id,))
        row = cursor.fetchone()
        
        if row:
            stored_faithfulness, stored_query = row
            print(f"Stored faithfulness: {stored_faithfulness}")
            print(f"Query stored: {stored_query[:50]}...")
        
        conn.close()
        print("-" * 50)

if __name__ == "__main__":
    inspect_evaluation_call()
