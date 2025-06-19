@echo off
REM Multi-RAG Chatbot Launcher - Windows Batch File
REM Double-click this file to launch the chatbot

echo ===============================================
echo 🤖 Multi-RAG Chatbot Launcher
echo ===============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    echo Download from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if we're in the right directory
if not exist "chatbot_app.py" (
    echo ❌ chatbot_app.py not found!
    echo Please run this batch file from the directory containing chatbot_app.py
    echo.
    pause
    exit /b 1
)

echo ✅ Chatbot files found
echo.

REM Check if .env file exists
if not exist ".env" (
    echo ❌ .env file not found!
    echo Creating .env template...
    echo GOOGLE_API_KEY=your_google_api_key_here > .env
    echo.
    echo ✅ .env file created
    echo Please edit .env file and add your Google API key
    echo Get your API key from: https://makersuite.google.com/app/apikey
    echo.
    echo Opening .env file for editing...
    notepad .env
    echo.
    echo Press any key when you've added your API key...
    pause >nul
)

echo ✅ .env file found
echo.

REM Install/check requirements
echo 🔧 Checking requirements...
pip install -q streamlit langchain python-dotenv >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Installing requirements...
    pip install -r requirements_chatbot.txt
    if errorlevel 1 (
        echo ❌ Error installing requirements
        echo Please run: pip install -r requirements_chatbot.txt
        pause
        exit /b 1
    )
)

echo ✅ Requirements satisfied
echo.

REM Launch options
echo 🚀 Choose your chatbot version:
echo.
echo 1. Basic Chatbot (recommended for first time)
echo 2. Advanced Chatbot with Comparisons
echo 3. Command Line Demo
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Launching Basic Chatbot...
    echo The chatbot will open in your web browser
    echo Press Ctrl+C in this window to stop the chatbot
    echo.
    streamlit run chatbot_app.py
    goto :end
)

if "%choice%"=="2" (
    echo.
    echo 🚀 Launching Advanced Chatbot...
    echo The chatbot will open in your web browser
    echo Press Ctrl+C in this window to stop the chatbot
    echo.
    streamlit run advanced_chatbot_app.py
    goto :end
)

if "%choice%"=="3" (
    echo.
    echo 🚀 Launching Command Line Demo...
    echo.
    python demo.py
    goto :end
)

if "%choice%"=="4" (
    echo 👋 Goodbye!
    goto :end
)

echo Invalid choice. Please run the script again and choose 1, 2, 3, or 4.

:end
echo.
echo Press any key to exit...
pause >nul
