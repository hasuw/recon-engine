@echo off
echo Starting deployment process...

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

:: Set environment variables
set FLASK_APP=wsgi.py
set FLASK_ENV=production

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Initialize database
echo Initializing database...
python db_init.py

:: Run the application
echo Starting application...
python wsgi.py
:: Check for virtual environment
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate
) else (
    echo Activating virtual environment...
    call .venv\Scripts\activate
)

echo Deployment completed successfully!

:: Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('stopwords')"

:: Check if gunicorn is installed
pip show gunicorn >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo gunicorn is not installed. Installing...
    pip install gunicorn
)

:: Start the application
echo Starting the application...
python wsgi.py

echo Deployment complete!