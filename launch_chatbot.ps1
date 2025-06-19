# Multi-RAG Chatbot Launcher for Windows
# Run this script to start the chatbot application

Write-Host "=" * 60 -ForegroundColor Blue
Write-Host "🤖 Multi-RAG Chatbot Launcher" -ForegroundColor Blue
Write-Host "=" * 60 -ForegroundColor Blue

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python first." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "chatbot_app.py")) {
    Write-Host "❌ chatbot_app.py not found in current directory" -ForegroundColor Red
    Write-Host "Please run this script from the directory containing chatbot_app.py" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "❌ .env file not found!" -ForegroundColor Red
    Write-Host "Creating .env file template..." -ForegroundColor Yellow
    
    $envContent = @"
# Replace 'your_google_api_key_here' with your actual Google API key
# You can get your API key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_google_api_key_here
"@
    
    $envContent | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "✅ .env file created. Please edit it and add your GOOGLE_API_KEY" -ForegroundColor Green
    Write-Host "Get your API key from: https://makersuite.google.com/app/apikey" -ForegroundColor Yellow
    
    # Open .env file for editing
    try {
        Start-Process notepad ".env"
    } catch {
        Write-Host "Please open .env file and add your API key" -ForegroundColor Yellow
    }
    
    Read-Host "Press Enter when you've added your API key to .env file"
}

# Check if required packages are installed
Write-Host "🔧 Checking required packages..." -ForegroundColor Cyan

$requiredPackages = @("streamlit", "langchain", "faiss-cpu", "python-dotenv")
$missingPackages = @()

foreach ($package in $requiredPackages) {
    try {
        python -c "import $($package.Replace('-', '_'))" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            $missingPackages += $package
        }
    } catch {
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "❌ Missing packages: $($missingPackages -join ', ')" -ForegroundColor Red
    
    $response = Read-Host "Install missing packages? (y/n)"
    if ($response.ToLower() -eq 'y') {
        Write-Host "📦 Installing packages..." -ForegroundColor Cyan
        
        try {
            pip install -r requirements_chatbot.txt
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ Packages installed successfully!" -ForegroundColor Green
            } else {
                Write-Host "❌ Error installing packages" -ForegroundColor Red
                Read-Host "Press Enter to exit"
                exit 1
            }
        } catch {
            Write-Host "❌ Error installing packages: $_" -ForegroundColor Red
            Read-Host "Press Enter to exit"
            exit 1
        }
    } else {
        Write-Host "Please install the required packages first:" -ForegroundColor Yellow
        Write-Host "pip install -r requirements_chatbot.txt" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "✅ All required packages are installed" -ForegroundColor Green
}

# Launch the chatbot
Write-Host "🚀 Launching Multi-RAG Chatbot..." -ForegroundColor Green
Write-Host "The chatbot will open in your web browser at http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow

try {
    streamlit run chatbot_app.py --server.address localhost --server.port 8501
} catch {
    Write-Host "❌ Error launching chatbot: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "👋 Chatbot stopped" -ForegroundColor Green
Read-Host "Press Enter to exit"
