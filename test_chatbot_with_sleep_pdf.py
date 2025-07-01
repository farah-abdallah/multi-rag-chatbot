#!/usr/bin/env python3
"""
Test the chatbot with the sleep PDF to verify the fixes
"""

import streamlit as st
import os
import sys

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_sleep_pdf():
    print("="*60)
    print("🧪 TESTING CHATBOT WITH SLEEP PDF")
    print("="*60)
    
    # Look for sleep PDF
    sleep_pdf_locations = [
        r"C:\Users\iTECH\Downloads\The_Importance_of_Sleep (1).pdf",
        r"c:\Users\iTECH\Downloads\The_Importance_of_Sleep.pdf",
        "./data/The_Importance_of_Sleep.pdf",
        "./The_Importance_of_Sleep.pdf"
    ]
    
    sleep_pdf = None
    for location in sleep_pdf_locations:
        if os.path.exists(location):
            sleep_pdf = location
            print(f"✅ Found sleep PDF: {location}")
            break
    
    if not sleep_pdf:
        print("❌ Sleep PDF not found in expected locations:")
        for loc in sleep_pdf_locations:
            print(f"   - {loc}")
        print("\n💡 To test properly:")
        print("   1. Save the Sleep PDF to your Downloads folder")
        print("   2. Or run the main chatbot and upload the PDF")
        print("   3. Ask: 'What are two cognitive effects of sleep deprivation?'")
        print("   4. Check the terminal output for debugging info")
        print("   5. Verify only relevant chunks are highlighted")
        return
    
    print(f"\n📋 TO TEST THE FIXES:")
    print(f"1. Run: streamlit run chatbot_app.py")
    print(f"2. Upload: {os.path.basename(sleep_pdf)}")
    print(f"3. Select: CRAG method")
    print(f"4. Ask: 'What are two cognitive effects of sleep deprivation?'")
    print(f"5. Check terminal for debugging output")
    print(f"6. Open document viewer and verify:")
    print(f"   - Only cognitive-related chunks are highlighted")
    print(f"   - Health benefits chunks have low scores")
    print(f"   - Page numbers are correct")
    
    print(f"\n🔍 EXPECTED IMPROVEMENTS:")
    print(f"✅ Page numbers should be correct (not 1, 2 for chunks on pages 2, 3)")
    print(f"✅ 'Health Benefits' chunk should get low score (<0.5)")
    print(f"✅ 'Cognitive Impairment' chunk should get high score (>0.7)")
    print(f"✅ Only relevant chunks should be highlighted in document viewer")
    print(f"✅ Terminal should show detailed debugging output")

if __name__ == "__main__":
    test_sleep_pdf()
