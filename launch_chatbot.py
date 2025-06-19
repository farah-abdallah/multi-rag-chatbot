#!/usr/bin/env python3
"""
Launcher script for Multi-RAG Chatbot Application

This script helps launch the Streamlit chatbot application with proper setup.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'langchain',
        'faiss_cpu',
        'python_dotenv',
        'google.generativeai'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').replace('_cpu', ''))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements_chatbot.txt'
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_env_file():
    """Check if .env file exists with API key"""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please create a .env file with your GOOGLE_API_KEY:")
        print("GOOGLE_API_KEY=your_api_key_here")
        print("\nYou can get your API key from: https://makersuite.google.com/app/apikey")
        return False
    
    # Check if API key is set
    with open(env_file, 'r') as f:
        content = f.read()
        if 'GOOGLE_API_KEY=' not in content or 'your_google_api_key_here' in content:
            print("⚠️ Please set your GOOGLE_API_KEY in the .env file")
            print("Get your API key from: https://makersuite.google.com/app/apikey")
            return False
    
    print("✅ .env file found with API key")
    return True

def launch_chatbot():
    """Launch the Streamlit chatbot application"""
    print("🚀 Launching Multi-RAG Chatbot...")
    print("The chatbot will open in your web browser at http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'chatbot_app.py',
            '--server.address', 'localhost',
            '--server.port', '8501',
            '--browser.serverAddress', 'localhost'
        ])
    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped by user")
    except Exception as e:
        print(f"❌ Error launching chatbot: {e}")

def main():
    """Main launcher function"""
    print("=" * 60)
    print("🤖 Multi-RAG Chatbot Launcher")
    print("=" * 60)
    
    # Check current directory
    if not Path('chatbot_app.py').exists():
        print("❌ chatbot_app.py not found in current directory")
        print("Please run this script from the directory containing chatbot_app.py")
        sys.exit(1)
    
    # Check requirements
    missing_packages = check_requirements()
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        
        response = input("Install missing packages? (y/n): ").lower().strip()
        if response == 'y':
            if not install_requirements():
                sys.exit(1)
        else:
            print("Please install the required packages first:")
            print("pip install -r requirements_chatbot.txt")
            sys.exit(1)
    else:
        print("✅ All required packages are installed")
    
    # Check environment file
    if not check_env_file():
        sys.exit(1)
    
    # Launch the chatbot
    launch_chatbot()

if __name__ == "__main__":
    main()
