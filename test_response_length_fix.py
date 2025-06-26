"""
Test script to verify response length calculation fix
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation_framework import EvaluationManager

def test_response_length_fix():
    """Test that response length is now calculated correctly"""
    
    print("🧪 Testing response length calculation fix...")
    
    # Create evaluation manager
    eval_manager = EvaluationManager("test_response_length.db")
    
    # Test case with actual response content
    test_query = "What is climate change?"
    test_response = "Climate change refers to long-term shifts in global temperatures and weather patterns. These changes may be natural, such as through variations in the solar cycle. But since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil and gas."
    test_technique = "Test Technique"
    test_context = "Some context about climate change and global warming effects."
    
    print(f"📝 Test response length: {len(test_response)} characters")
    print(f"📝 Test response word count: {len(test_response.split())} words")
    
    # Evaluate the response
    query_id = eval_manager.evaluate_rag_response(
        query=test_query,
        response=test_response,
        technique=test_technique,
        document_sources=["test.txt"],
        context=test_context,
        processing_time=1.5
    )
    
    # Check what was stored in database
    import sqlite3
    conn = sqlite3.connect("test_response_length.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT response_length, faithfulness_score, relevance_score, completeness_score, processing_time 
        FROM evaluations WHERE query_id = ?
    """, (query_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        stored_length, faithfulness, relevance, completeness, proc_time = row
        print(f"\n✅ Database Results:")
        print(f"   📏 Stored response length: {stored_length}")
        print(f"   🎯 Faithfulness score: {faithfulness}")
        print(f"   📊 Relevance score: {relevance}")
        print(f"   ✅ Completeness score: {completeness}")
        print(f"   ⏱️ Processing time: {proc_time}s")
        
        # Verify the fix
        expected_length = len(test_response)
        if stored_length == expected_length:
            print(f"\n🎉 SUCCESS: Response length is correctly calculated!")
            print(f"   Expected: {expected_length}, Got: {stored_length}")
        else:
            print(f"\n❌ FAILED: Response length mismatch!")
            print(f"   Expected: {expected_length}, Got: {stored_length}")
            
        if faithfulness > 0:
            print(f"✅ Faithfulness is working: {faithfulness}")
        else:
            print(f"❌ Faithfulness is still 0: {faithfulness}")
            
    else:
        print("❌ No data found in database!")
    
    # Clean up test database
    try:
        os.remove("test_response_length.db")
        print("\n🧹 Test database cleaned up")
    except:
        pass

if __name__ == "__main__":
    test_response_length_fix()
