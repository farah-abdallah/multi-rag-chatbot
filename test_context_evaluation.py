"""
Test the improved evaluation framework with proper context and varied scoring
"""

import os
from evaluation_framework import EvaluationManager, UserFeedback

def test_improved_evaluation_with_context():
    print("🧪 Testing Improved Evaluation with Context")
    print("=" * 60)
    
    # Initialize evaluation manager
    manager = EvaluationManager("test_improved_context.db")
    print("1. ✅ Evaluation manager initialized")
    
    # Test scenarios with different contexts and responses
    test_scenarios = [
        {
            "technique": "Basic RAG",
            "query": "What is climate change?",
            "response": "Climate change refers to long-term changes in global or regional climate patterns. It is primarily caused by human activities, especially the emission of greenhouse gases like carbon dioxide from burning fossil fuels. The effects include rising temperatures, melting ice caps, and extreme weather events.",
            "context": "Climate change is a phenomenon where Earth's climate system undergoes long-term alterations. The primary driver is increased greenhouse gas concentrations in the atmosphere due to human activities. These gases trap heat, leading to global warming. Scientists have documented rising sea levels, changing precipitation patterns, and more frequent extreme weather events as consequences of climate change."
        },
        {
            "technique": "Explainable Retrieval",
            "query": "How does deforestation affect the environment?",
            "response": "Deforestation has several negative environmental impacts including habitat loss, reduced carbon absorption, soil erosion, and disruption of water cycles.",
            "context": "Forests play a crucial role in environmental balance. They absorb carbon dioxide during photosynthesis, provide habitat for countless species, prevent soil erosion with their root systems, and regulate local water cycles through transpiration. When forests are cleared, these benefits are lost, leading to increased atmospheric CO2, biodiversity loss, degraded soil quality, and altered precipitation patterns."
        },
        {
            "technique": "CRAG",
            "query": "What are renewable energy sources?",
            "response": "Based on current research, renewable energy includes solar power from photovoltaic panels, wind energy from turbines, hydroelectric power from dams, geothermal energy from Earth's heat, and biomass energy from organic materials. These sources are sustainable because they naturally replenish and produce minimal greenhouse gas emissions.",
            "context": "Renewable energy sources are those that can be replenished naturally and sustainably. Solar energy harnesses sunlight through photovoltaic cells or thermal collectors. Wind energy captures kinetic energy from moving air using turbines. Hydroelectric power utilizes flowing water to generate electricity. Geothermal energy taps into Earth's internal heat. Biomass converts organic matter into energy through combustion or biochemical processes."
        },
        {
            "technique": "Document Augmentation",
            "query": "What causes air pollution?",
            "response": "Air pollution comes from cars.",  # Incomplete response
            "context": "Air pollution is caused by various sources including vehicle emissions (nitrogen oxides, carbon monoxide), industrial processes (sulfur dioxide, particulate matter), power generation (coal burning), agricultural activities (ammonia, methane), and natural sources (dust storms, volcanic eruptions). The main pollutants include particulate matter, ozone, nitrogen dioxide, sulfur dioxide, and carbon monoxide."
        },
        {
            "technique": "Adaptive RAG",
            "query": "Explain the water cycle",
            "response": "The water cycle is the continuous movement of water on, above, and below Earth's surface. It begins with evaporation from oceans, lakes, and rivers, where solar energy converts liquid water to vapor. This water vapor rises into the atmosphere where it cools and condenses into clouds through condensation. When clouds become saturated, precipitation occurs as rain, snow, or other forms. The precipitated water then flows back to water bodies through surface runoff or infiltrates into the ground as groundwater, eventually returning to the oceans to complete the cycle.",
            "context": "The water cycle describes the continuous circulation of water throughout Earth's systems. Solar radiation drives evaporation from surface water bodies. Water vapor rises and undergoes condensation to form clouds. Precipitation returns water to Earth's surface. Some water flows as surface runoff to rivers and oceans, while other water infiltrates soil to become groundwater. Plants also contribute through transpiration, releasing water vapor from their leaves."
        }
    ]
    
    print("\n2. 🔄 Testing evaluation scenarios...")
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"   Scenario {i}: {scenario['technique']}")
        
        # Evaluate the response
        query_id = manager.evaluate_rag_response(
            query=scenario["query"],
            response=scenario["response"],
            technique=scenario["technique"],
            document_sources=["test_document.pdf"],
            context=scenario["context"],
            processing_time=1.0 + i * 0.3,
            session_id="test_session"
        )
        
        # Add user feedback
        feedback = UserFeedback(
            helpfulness=5 - (i % 3),  # Vary feedback: 5, 4, 3, 2, 5
            accuracy=4 if i != 4 else 2,  # Lower accuracy for incomplete response
            clarity=5 - (i % 4),
            overall_rating=5 - (i % 3),
            comments=f"Test feedback for {scenario['technique']}",
            timestamp="2025-06-20T10:00:00"
        )
        
        manager.add_user_feedback(query_id, feedback)
        results.append(query_id)
        print(f"      ✅ Evaluated and stored with ID: {query_id[:8]}...")
    
    print("\n3. 📊 Analyzing results...")
    comparison_data = manager.get_technique_comparison()
    
    print("\n📈 Evaluation Results:")
    print("-" * 80)
    print(f"{'Technique':<20} {'Relevance':<10} {'Faithful':<10} {'Complete':<10} {'Semantic':<10} {'Rating':<8}")
    print("-" * 80)
    
    for technique, data in comparison_data.items():
        print(f"{technique:<20} {data['avg_relevance']:<10.3f} {data['avg_faithfulness']:<10.3f} {data['avg_completeness']:<10.3f} {data['avg_semantic_similarity']:<10.3f} {data['avg_user_rating']:<8.1f}")
    
    print(f"\n✅ Test completed! Database: test_improved_context.db")
    
    # Check if we have more varied scores
    relevance_scores = [data['avg_relevance'] for data in comparison_data.values()]
    faithfulness_scores = [data['avg_faithfulness'] for data in comparison_data.values()]
    completeness_scores = [data['avg_completeness'] for data in comparison_data.values()]
    
    print(f"\n🔍 Score Analysis:")
    print(f"   Relevance variation: {max(relevance_scores) - min(relevance_scores):.3f}")
    print(f"   Faithfulness variation: {max(faithfulness_scores) - min(faithfulness_scores):.3f}")
    print(f"   Completeness variation: {max(completeness_scores) - min(completeness_scores):.3f}")
    
    if max(faithfulness_scores) > 0.1:
        print("   ✅ Faithfulness scores are now varied (context is working!)")
    else:
        print("   ⚠️ Faithfulness scores still low (context might need adjustment)")
    
    if max(completeness_scores) - min(completeness_scores) > 0.2:
        print("   ✅ Completeness scores show good variation")
    else:
        print("   ⚠️ Completeness scores need more variation")

if __name__ == "__main__":
    test_improved_evaluation_with_context()
