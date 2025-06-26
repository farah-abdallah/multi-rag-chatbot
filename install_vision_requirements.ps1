# Vision-Enhanced Multi-RAG Installation Script for Windows
# Run with: powershell -ExecutionPolicy Bypass -File install_vision_requirements.ps1

Write-Host "🔍 Installing Vision-Enhanced Multi-RAG Dependencies..." -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python first." -ForegroundColor Red
    exit 1
}

# Check if pip is available
try {
    pip --version | Out-Null
    Write-Host "✅ pip is available" -ForegroundColor Green
} catch {
    Write-Host "❌ pip not found. Please install pip first." -ForegroundColor Red
    exit 1
}

# Upgrade pip first
Write-Host "📦 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install core requirements
Write-Host "📦 Installing core Multi-RAG dependencies..." -ForegroundColor Yellow
pip install langchain langchain-google-genai langchain-community faiss-cpu python-dotenv sentence-transformers streamlit pandas

# Install vision processing dependencies
Write-Host "🔍 Installing vision processing dependencies..." -ForegroundColor Yellow
pip install PyMuPDF Pillow

# Install optional enhanced processing
Write-Host "📄 Installing enhanced document processing..." -ForegroundColor Yellow
pip install unstructured

# Install web search dependencies
Write-Host "🌐 Installing web search dependencies..." -ForegroundColor Yellow
pip install duckduckgo-search requests

# Verify installations
Write-Host "✅ Verifying installations..." -ForegroundColor Yellow

$packages = @(
    "langchain",
    "langchain-google-genai", 
    "langchain-community",
    "faiss-cpu",
    "python-dotenv",
    "sentence-transformers",
    "streamlit",
    "pandas",
    "PyMuPDF",
    "Pillow",
    "unstructured",
    "duckduckgo-search",
    "requests"
)

foreach ($package in $packages) {
    try {
        $version = pip show $package 2>$null | Select-String "Version:" | ForEach-Object { $_.ToString().Split(":")[1].Trim() }
        if ($version) {
            Write-Host "✅ $package $version" -ForegroundColor Green
        } else {
            Write-Host "⚠️ $package not found" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ Error checking $package" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🎉 Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "1. Set your GOOGLE_API_KEY environment variable" -ForegroundColor White
Write-Host "2. Run the chatbot with: streamlit run vision_enhanced_chatbot.py" -ForegroundColor White
Write-Host "3. Upload documents and start chatting!" -ForegroundColor White
Write-Host ""
Write-Host "🔍 Vision features enabled for:" -ForegroundColor Cyan
Write-Host "   • PDF image extraction and analysis" -ForegroundColor White
Write-Host "   • Chart and graph data extraction" -ForegroundColor White
Write-Host "   • Diagram and infographic understanding" -ForegroundColor White
Write-Host "   • Visual content integration with text" -ForegroundColor White
