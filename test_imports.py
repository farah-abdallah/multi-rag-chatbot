"""
Quick test to check if the Streamlit app can be imported without errors
"""

try:
    print("🧪 Testing imports...")
    
    # Test evaluation framework
    from evaluation_framework import EvaluationManager, UserFeedback, EvaluationMetrics
    print("✅ Evaluation framework imported successfully")
    
    # Test analytics dashboard
    from analytics_dashboard import display_analytics_dashboard
    print("✅ Analytics dashboard imported successfully")
    
    # Test main chatbot app components
    import chatbot_app
    print("✅ Chatbot app imported successfully")
    
    print("\n🎉 All imports successful! The app should work without errors.")
    print("\n🚀 To run the app, use: streamlit run chatbot_app.py")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
