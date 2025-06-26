"""
Demo script to test the evaluation framework functionality
"""

import os
import sys
from evaluation_framework import EvaluationManager, UserFeedback
from datetime import datetime

def demo_evaluation_framework():
    """Demonstrate the evaluation framework capabilities"""
    print("🧪 Testing RAG Evaluation Framework")
    print("=" * 50)
    
    # Initialize evaluation manager
    print("1. Initializing evaluation manager...")
    eval_manager = EvaluationManager("demo_evaluation.db")
    
    # Simulate some RAG responses for testing
    print("\n2. Simulating RAG responses...")
    
    test_scenarios = [
        {
            "query": "What is climate change?",
            "response": "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, scientific evidence shows that human activities, particularly burning fossil fuels, have been the main driver of climate change since the 1800s.",
            "technique": "Adaptive RAG",
            "documents": ["Understanding_Climate_Change.pdf"]
        },
        {
            "query": "How does AI work?",
            "response": "Artificial Intelligence (AI) works by using algorithms and computational models to simulate human intelligence. Machine learning, a subset of AI, enables systems to learn and improve from experience without being explicitly programmed.",
            "technique": "CRAG",
            "documents": ["ai_basics.pdf"]
        },
        {
            "query": "What are renewable energy sources?",
            "response": "Renewable energy sources include solar, wind, hydroelectric, geothermal, and biomass energy. These sources naturally replenish themselves and produce minimal environmental impact compared to fossil fuels.",
            "technique": "Explainable Retrieval",
            "documents": ["energy_report.pdf"]
        }
    ]
    
    query_ids = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"   Evaluating scenario {i}: {scenario['technique']}")
        
        # Evaluate the response
        query_id = eval_manager.evaluate_rag_response(
            query=scenario["query"],
            response=scenario["response"],
            technique=scenario["technique"],
            document_sources=scenario["documents"],
            context="Sample context for evaluation",
            processing_time=1.5,
            session_id="demo_session"
        )
        
        query_ids.append(query_id)
        print(f"   ✅ Stored evaluation with ID: {query_id[:8]}...")
    
    # Simulate user feedback
    print("\n3. Simulating user feedback...")
    
    feedback_examples = [
        UserFeedback(
            helpfulness=5,
            accuracy=4,
            clarity=5,
            overall_rating=5,
            comments="Excellent explanation of climate change!",
            timestamp=datetime.now().isoformat()
        ),
        UserFeedback(
            helpfulness=4,
            accuracy=3,
            clarity=4,
            overall_rating=4,
            comments="Good AI explanation, could be more detailed.",
            timestamp=datetime.now().isoformat()
        ),
        UserFeedback(
            helpfulness=5,
            accuracy=5,
            clarity=4,
            overall_rating=5,
            comments="Very helpful renewable energy overview.",
            timestamp=datetime.now().isoformat()
        )
    ]
    
    for query_id, feedback in zip(query_ids, feedback_examples):
        eval_manager.add_user_feedback(query_id, feedback)
        print(f"   ✅ Added feedback for query {query_id[:8]}...")
    
    # Get comparison data
    print("\n4. Generating performance comparison...")
    comparison_data = eval_manager.get_technique_comparison()
    
    print("\n📊 Performance Summary:")
    print("-" * 30)
    
    for technique, data in comparison_data.items():
        print(f"\n{technique}:")
        print(f"  • Total queries: {data['total_queries']}")
        print(f"  • Average relevance: {data['avg_relevance']:.3f}")
        print(f"  • Average user rating: {data['avg_user_rating']:.1f}/5")
        print(f"  • Processing time: {data['avg_processing_time']:.2f}s")
        print(f"  • Feedback count: {data['feedback_count']}")
    
    # Export data
    print("\n5. Exporting evaluation data...")
    export_file = eval_manager.export_evaluation_data("demo_export.json")
    print(f"   ✅ Data exported to: {export_file}")
    
    print("\n🎉 Evaluation framework demo completed successfully!")
    print(f"📁 Demo database created: demo_evaluation.db")
    print(f"📁 Export file created: {export_file}")
    
    return True

if __name__ == "__main__":
    try:
        demo_evaluation_framework()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
