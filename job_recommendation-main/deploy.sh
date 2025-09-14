#!/bin/bash

# Simple deployment script for the Internship Recommender application

echo "Starting deployment process..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip is not installed. Please install pip and try again."
    exit 1
fi

# Set environment variables
export FLASK_APP=wsgi.py
export FLASK_ENV=production

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
else
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Initialize database
echo "Initializing database..."
python db_init.py

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('stopwords')"

# Check if gunicorn is installed
if ! command -v gunicorn &> /dev/null; then
    echo "gunicorn is not installed. Installing..."
    pip install gunicorn
fi

# Start the application
echo "Starting the application..."
gunicorn wsgi:app --bind 0.0.0.0:8080

echo "Deployment complete!"