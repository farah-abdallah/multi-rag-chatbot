"""
Utility script to initialize the project and check dependencies.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path


def check_python_version():
    """Check if Python version is supported."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Python 3.8+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is supported.")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found.")
        return False
    
    missing_packages = []
    
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                package_name = line.split('==')[0].split('>=')[0].split('<=')[0]
                if not is_package_installed(package_name):
                    missing_packages.append(line)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed.")
    return True


def is_package_installed(package_name):
    """Check if a package is installed."""
    # Handle package name mapping
    package_mapping = {
        'python-docx': 'docx',
        'python-dotenv': 'dotenv',
        'duckduckgo-search': 'duckduckgo_search',
        'google-generativeai': 'google.generativeai'
    }
    
    actual_name = package_mapping.get(package_name, package_name)
    
    try:
        importlib.util.find_spec(actual_name)
        return True
    except ImportError:
        return False


def check_environment_variables():
    """Check if required environment variables are set."""
    required_vars = ['GEMINI_API_KEYS']
    optional_vars = ['GOOGLE_SEARCH_API_KEY', 'GOOGLE_SEARCH_ENGINE_ID']
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    for var in optional_vars:
        if not os.getenv(var):
            missing_optional.append(var)
    
    if missing_required:
        print(f"❌ Missing required environment variables: {', '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"⚠️  Missing optional environment variables: {', '.join(missing_optional)}")
    
    print("✅ Required environment variables are set.")
    return True


def create_directories():
    """Create necessary directories."""
    directories = ['logs', 'data', 'data/sample_documents']
    project_root = Path(__file__).parent.parent
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def install_dependencies():
    """Install required dependencies."""
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found.")
        return False
    
    try:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        print("✅ Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def create_env_file():
    """Create .env file from template."""
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    if env_file.exists():
        print("✅ .env file already exists.")
        return True
    
    if env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template.")
        print("⚠️  Please edit .env file with your API keys.")
        return True
    
    # Create basic .env file
    env_content = """# Gemini API Keys (comma-separated for rotation)
GEMINI_API_KEYS=your_api_key_1,your_api_key_2,your_api_key_3

# Google Search API (optional, for web search)
GOOGLE_SEARCH_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id

# Application Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_RETRIEVAL_RESULTS=5
MAX_WEB_RESULTS=3
LLM_TEMPERATURE=0.7
LLM_MODEL_NAME=gemini-pro

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file with template.")
    print("⚠️  Please edit .env file with your API keys.")
    return True


def main():
    """Main setup function."""
    print("🚀 Setting up Multi-RAG Chatbot...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Check if dependencies need to be installed
    if not check_dependencies():
        response = input("Install missing dependencies? (y/n): ").lower()
        if response == 'y':
            if not install_dependencies():
                sys.exit(1)
        else:
            print("❌ Dependencies not installed. Please install them manually.")
            sys.exit(1)
    
    # Check environment variables
    if not check_environment_variables():
        print("⚠️  Please set the required environment variables in .env file.")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: streamlit run app.py")
    print("3. Or try CLI: python cli.py --help")


if __name__ == "__main__":
    main()
