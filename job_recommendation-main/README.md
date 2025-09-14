# AI-Powered Job Recommender (Flask)

## Overview
Production-ready Flask application that analyzes your resume and recommends relevant jobs using NLP (NLTK + TF‑IDF + cosine similarity). Upload a PDF/TXT resume and get tailored job matches with links to apply. Includes REST API, CSV export, error handling, logging, and cloud deployment configs.

## Features
- **Resume Analysis**: Upload PDF/TXT resume for analysis
- **Job Matching**: Personalized job recommendations using TF‑IDF + cosine similarity
- **Keyword Extraction**: Top keywords from your resume
- **REST API**: Programmatic access to recommendations
- **CSV Export**: Download recommendations as CSV
- **Operational Hardening**: Graceful error handling, secure temp file deletion, structured logging

## Tech Stack
- **Backend**: Flask
- **NLP**: NLTK, scikit-learn (TF‑IDF, cosine similarity)
- **Data**: pandas, numpy
- **Docs**: PyPDF2
- **Scraping**: BeautifulSoup, Selenium (optional; app uses generated sample data by default)

## How It Works

### For Users
1. Upload your resume in PDF or TXT format
2. The system analyzes your resume using NLP techniques
3. View recommended internship postings based on your skills and experience
4. Download results or apply directly through provided links

### Behind the Scenes
- **Data Collection**: Web scraping using Selenium and BeautifulSoup to gather job postings
- **Text Processing**: Removing stopwords, lemmatization, and keeping only alphabetic characters
- **Machine Learning**: TF-IDF vectorization and cosine similarity to match resume content with job descriptions

## Local Development
1. Python 3.9+ recommended (runtime: `python-3.9.18`).
2. Create and activate a virtualenv.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   python app.py
   ```
5. Open http://localhost:8080

Notes:
- On first run, the app downloads required NLTK corpora.
- Uploads are stored temporarily and securely wiped after processing.

## API Documentation

### POST /api/recommend
Multipart form fields:
- `file`: resume file (PDF/TXT)
- `job_domain`: e.g. `software_engineering`, `data_science`

Response shape:
```json
{
  "filename": "resume.pdf",
  "job_domain": "software_engineering",
  "top_keywords": ["python", "flask", "ml"],
  "recommendations": [
    {
      "Company": "Example Corp",
      "Role": "Software Engineer",
      "Location": "Bangalore",
      "Date Posted": "2025-01-01",
      "Application/Link": "https://example.com/job/123",
      "relevance_score": 92
    }
  ]
}
```

### GET /api/job_domains
Returns human-friendly list of configured job domains.

### GET /download_csv
Downloads the current recommendations as CSV (after an upload session).

## Deployment Options

### Option 1: Deploy to Render
1. Fork/clone this repository to your GitHub account
2. Sign up for a [Render](https://render.com/) account
3. Create a new Web Service and connect your GitHub repository
4. Select the Python environment
5. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn wsgi:app`
6. Set up environment variables:
   - `DATABASE_URL`: Your PostgreSQL database URL
   - `SECRET_KEY`: A secure random string for session encryption
   - `FLASK_ENV`: Set to `production` for deployment
7. Deploy the application

### Option 2: Deploy with Docker (local) / Google Cloud Run
1. Make sure Docker is installed on your system
2. Create a `.env` file with your environment variables:
   ```
   DATABASE_URL=postgresql://username:password@host:port/database
   SECRET_KEY=your_secure_random_string
   FLASK_ENV=production
   ```
3. Build the Docker image:
   ```
   docker build -t internship-recommender .
   ```
4. Run the container locally:
   ```
   docker run -p 8080:8080 --env-file .env internship-recommender
   ```
5. Access the application at http://localhost:8080

### Option 3: Deploy to Heroku
1. Make sure you have the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) installed
2. Login to Heroku:
   ```
   heroku login
   ```
3. Create a new Heroku app:
   ```
   heroku create your-app-name
   ```
4. Add a PostgreSQL database:
   ```
   heroku addons:create heroku-postgresql:hobby-dev
   ```
5. Set environment variables:
   ```
   heroku config:set SECRET_KEY=your_secure_random_string
   heroku config:set FLASK_ENV=production
   ```
6. Deploy the application:
   ```
   git push heroku main
   ```
7. Open the application:
   ```
   heroku open
   ```
### Option 4: Deploy to Google Cloud Run
1. Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
2. Initialize and authenticate:
   ```
   gcloud init
   gcloud auth login
   ```
3. Build and deploy the container:
   ```
   gcloud builds submit --tag gcr.io/[YOUR_PROJECT_ID]/internship-recommender
   gcloud run deploy internship-recommender --image gcr.io/[YOUR_PROJECT_ID]/internship-recommender --platform managed --allow-unauthenticated
   ```
4. Set environment variables in the Google Cloud Console

## CLI Examples
```bash
# API call
curl -X POST -F "file=@resume.pdf" -F "job_domain=software_engineering" http://localhost:8080/api/recommend
```

## Testing
Run unit tests:
```bash
python -m unittest -v
```

## Author
zerospectre





