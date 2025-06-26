#!/usr/bin/env python3

"""
Test script to verify the improved faithfulness evaluation
"""

def test_faithfulness():
    print("🧪 Testing improved faithfulness evaluation...")
    
    try:
        from evaluation_framework import AutomatedEvaluator
          # Initialize evaluator (it will automatically set use_llm_evaluation=True if Gemini is available)
        evaluator = AutomatedEvaluator()
        
        # Test with realistic context and responses
        context = """Climate change is primarily caused by greenhouse gas emissions from human activities such as burning fossil fuels, deforestation, and industrial processes. These gases trap heat in the atmosphere, leading to global warming. The main greenhouse gases include carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O). Scientific consensus shows that human activities have increased atmospheric CO2 levels by over 40% since pre-industrial times."""
        
        test_cases = [
            ("Greenhouse gases cause climate change.", "Basic correct response"),
            ("Climate change is caused by greenhouse gases from burning fossil fuels and deforestation, which trap heat in the atmosphere.", "Detailed correct response"),
            ("Human activities increase greenhouse gas emissions, leading to global warming through heat trapping.", "Comprehensive response"),
            ("Climate change happens sometimes.", "Vague, minimal detail"),
            ("Aliens are responsible for climate change through mind control.", "Completely incorrect"),
            ("The weather is changing.", "Too general, no specific connection")
        ]
        
        print("📊 Faithfulness evaluation results:")
        print("-" * 60)
        
        for response, description in test_cases:
            try:
                score = evaluator.evaluate_faithfulness(context, response)
                print(f"{score:.3f} | {description}")
                print(f"      | Response: {response}")
                print("-" * 60)
            except Exception as e:
                print(f"Error evaluating '{description}': {e}")
                print("-" * 60)
        
        print("✅ Faithfulness evaluation test completed!")
        
    except Exception as e:
        print(f"❌ Error during faithfulness test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_faithfulness()
