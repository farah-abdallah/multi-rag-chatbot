# Install required packages for Gemini-based RAG Chunk Size Evaluation
Write-Host "Installing required packages for chunk size evaluation..." -ForegroundColor Green

pip install llama-index google-generativeai nest-asyncio python-dotenv

Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "Make sure to set your GOOGLE_API_KEY environment variable or add it to a .env file" -ForegroundColor Yellow
Write-Host "Get your API key from: https://makersuite.google.com/app/apikey" -ForegroundColor Yellow
