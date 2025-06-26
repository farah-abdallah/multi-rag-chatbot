"""
Quick test to verify the faithfulness fix
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation_framework import EvaluationManager

def test_faithfulness_fix():
    """Test that faithfulness is no longer 0 for empty context"""
    
    print("🧪 Testing faithfulness evaluation fix...")
    
    # Create evaluation manager
    eval_manager = EvaluationManager("test_fix.db")
    
    # Test case with empty context (this was the problem)
    query = "What is climate change?"
    response = "Climate change refers to long-term shifts in global temperatures and weather patterns."
    context = ""  # Empty context - this was causing 0.0 faithfulness
    
    print(f"Query: {query}")
    print(f"Response: {response[:50]}...")
    print(f"Context: '{context}' (empty)")
    
    # Evaluate the response
    query_id = eval_manager.evaluate_rag_response(
        query=query,
        response=response,
        technique="Test Technique",
        document_sources=["test.txt"],
        context=context,  # Empty context
        processing_time=1.0,
        session_id="test_session"
    )
    
    # Check what was stored
    import sqlite3
    conn = sqlite3.connect("test_fix.db")
    cursor = conn.cursor()
    cursor.execute("SELECT faithfulness_score FROM evaluations WHERE query_id = ?", (query_id,))
    row = cursor.fetchone()
    
    if row:
        faithfulness_score = row[0]
        print(f"✅ Faithfulness score: {faithfulness_score}")
        
        if faithfulness_score > 0.0:
            print("🎉 SUCCESS: Faithfulness is no longer 0!")
            print(f"   Expected: > 0.0, Got: {faithfulness_score}")
        else:
            print("❌ FAILED: Faithfulness is still 0")
    else:
        print("❌ ERROR: Could not retrieve evaluation result")
    
    conn.close()
    
    # Clean up test database
    try:
        os.remove("test_fix.db")
    except:
        pass

if __name__ == "__main__":
    test_faithfulness_fix()
