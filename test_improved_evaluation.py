"""
Test the improved evaluation framework with heuristic fallbacks
"""

from evaluation_framework import EvaluationManager, UserFeedback
from datetime import datetime

def test_improved_evaluation():
    print("🧪 Testing Improved RAG Evaluation Framework")
    print("=" * 60)
    
    # Initialize evaluation manager
    print("1. Initializing evaluation manager...")
    manager = EvaluationManager(db_path="test_improved_evaluation.db")
    
    # Test scenarios with different characteristics
    test_scenarios = [
        {
            "query": "What is climate change?",
            "response": "Climate change refers to long-term shifts in global temperatures and weather patterns. It's primarily caused by human activities like burning fossil fuels.",
            "technique": "Adaptive RAG",
            "context": "Climate change is a complex phenomenon involving temperature increases, weather pattern changes, and environmental impacts."
        },
        {
            "query": "What is climate change?", 
            "response": "Yes.",  # Very short, incomplete response
            "technique": "CRAG",
            "context": "Climate change is a complex phenomenon involving temperature increases, weather pattern changes, and environmental impacts."
        },
        {
            "query": "What is climate change?",
            "response": "Climate change is a critical environmental issue that refers to significant and lasting changes in Earth's climate patterns. This phenomenon encompasses rising global temperatures, alterations in precipitation patterns, increased frequency of extreme weather events, and shifts in seasonal patterns. The primary driver of contemporary climate change is the enhanced greenhouse effect resulting from increased concentrations of greenhouse gases in the atmosphere, particularly carbon dioxide from fossil fuel combustion, deforestation, and industrial processes. The impacts include rising sea levels, melting ice caps, ecosystem disruptions, and threats to food security and human health.",
            "technique": "Explainable Retrieval",
            "context": "Climate change is a complex phenomenon involving temperature increases, weather pattern changes, and environmental impacts."
        },
        {
            "query": "What is climate change?",
            "response": "The weather today is sunny and nice. I like ice cream and pizza.",  # Irrelevant response
            "technique": "Document Augmentation",
            "context": "Climate change is a complex phenomenon involving temperature increases, weather pattern changes, and environmental impacts."
        },
        {
            "query": "Explain photosynthesis",
            "response": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.",
            "technique": "Basic RAG",
            "context": "Photosynthesis occurs in chloroplasts and involves light-dependent and light-independent reactions."
        }
    ]
    
    print("2. Testing different response scenarios...")
    query_ids = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"   Testing scenario {i}: {scenario['technique']}")
        
        query_id = manager.evaluate_rag_response(
            query=scenario["query"],
            response=scenario["response"],
            technique=scenario["technique"],
            document_sources=["test_document.pdf"],
            context=scenario["context"],
            processing_time=1.5,
            session_id="test_session"
        )
        query_ids.append(query_id)
        print(f"   ✅ Evaluated with ID: {query_id[:8]}...")
    
    print("\n3. Adding user feedback...")
    feedback_scores = [
        {"helpfulness": 5, "accuracy": 5, "clarity": 4, "overall": 5},  # Good response
        {"helpfulness": 1, "accuracy": 2, "clarity": 2, "overall": 1},  # Poor response
        {"helpfulness": 5, "accuracy": 5, "clarity": 5, "overall": 5},  # Excellent response
        {"helpfulness": 1, "accuracy": 1, "clarity": 1, "overall": 1},  # Terrible response
        {"helpfulness": 4, "accuracy": 4, "clarity": 4, "overall": 4},  # Good response
    ]
    
    for query_id, scores in zip(query_ids, feedback_scores):
        feedback = UserFeedback(
            helpfulness=scores["helpfulness"],
            accuracy=scores["accuracy"], 
            clarity=scores["clarity"],
            overall_rating=scores["overall"],
            comments=f"Test feedback for scenario",
            timestamp=datetime.now().isoformat()
        )
        manager.add_user_feedback(query_id, feedback)
        print(f"   ✅ Added feedback for {query_id[:8]}...")
    
    print("\n4. Analyzing results...")
    comparison_data = manager.get_technique_comparison()
    
    print("\n📊 Evaluation Results:")
    print("-" * 80)
    print(f"{'Technique':<20} {'Relevance':<10} {'Faithful':<10} {'Complete':<10} {'Semantic':<10} {'User Rating':<12}")
    print("-" * 80)
    
    for technique, data in comparison_data.items():
        relevance = data.get('avg_relevance', 0) or 0
        faithfulness = data.get('avg_faithfulness', 0) or 0 
        completeness = data.get('avg_completeness', 0) or 0
        semantic = data.get('avg_semantic_similarity', 0) or 0
        rating = data.get('avg_user_rating', 0) or 0
        
        print(f"{technique:<20} {relevance:<10.3f} {faithfulness:<10.3f} {completeness:<10.3f} {semantic:<10.3f} {rating:<12.1f}")
    
    print(f"\n🎉 Test completed! Database: test_improved_evaluation.db")
    print("\nKey observations:")
    print("- Relevance should vary based on how well responses address the query")
    print("- Faithfulness should reflect adherence to provided context")
    print("- Completeness should vary based on response depth and thoroughness")
    print("- Semantic similarity uses embeddings and should vary")
    print("- User ratings should reflect the manual scores we provided")

if __name__ == "__main__":
    test_improved_evaluation()
