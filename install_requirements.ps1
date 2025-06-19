# Install required packages for Gemini-based Adaptive RAG
Write-Host "Installing required packages..." -ForegroundColor Green

pip install langchain langchain-google-genai langchain-community faiss-cpu python-dotenv

Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "Make sure to set your GOOGLE_API_KEY environment variable or add it to a .env file" -ForegroundColor Yellow
