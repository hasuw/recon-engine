import os
import re
import uuid
import logging
import pandas as pd
import nltk
import PyPDF2
import json
import time
import threading
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, flash, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from io import BytesIO
from models import db, User, Resume, JobSearch, JobRecommendation, JobPortal, init_db, JobDomain
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up additional security logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)
security_logger = logging.getLogger('security')
security_logger.setLevel(logging.WARNING)

# Constants
NOUN_TAGS = ['NN', 'NNS', 'NNP', 'NNPS']
VERB_TAGS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
ADJ_TAGS = ['JJ', 'JJR', 'JJS']
ADV_TAGS = ['RB', 'RBR', 'RBS']
NUM_POSTINGS = 10
TOP_KEYWORDS = 10

# Job domains with sample Indian companies
JOB_DOMAINS = {
    'software_engineering': ['Infosys', 'TCS', 'Wipro', 'HCL Technologies', 'Tech Mahindra'],
    'data_science': ['Mu Sigma', 'Fractal Analytics', 'Tiger Analytics', 'Absolutdata', 'LatentView'],
    'web_development': ['Cognizant', 'Mindtree', 'Persistent Systems', 'Mphasis', 'LTIMindtree'],
    'finance': ['HDFC Bank', 'ICICI Bank', 'SBI', 'Axis Bank', 'Kotak Mahindra Bank'],
    'marketing': ['Reliance', 'Tata Group', 'Hindustan Unilever', 'ITC', 'Bharti Airtel'],
    'healthcare': ['Apollo Hospitals', 'Fortis Healthcare', 'Max Healthcare', 'Manipal Hospitals', 'Medanta'],
    'education': ['BYJU\'S', 'Unacademy', 'Vedantu', 'upGrad', 'Great Learning'],
    'ecommerce': ['Flipkart', 'Amazon India', 'Myntra', 'Snapdeal', 'Nykaa'],
    'startup': ['Zomato', 'Swiggy', 'Paytm', 'Ola', 'CRED']
}

# Indian cities for job locations
INDIAN_CITIES = [
    'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad', 
    'Jaipur', 'Lucknow', 'Kochi', 'Indore', 'Chandigarh', 'Coimbatore', 'Gurgaon', 'Noida'
]

# Create Flask application
def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    init_db(app)
    
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize rate limiter
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://",
    )
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register routes
    register_routes(app, limiter)
    
    return app

# Download NLTK resources
def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        logger.info("NLTK resources downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {e}")

# Register error handlers
def register_error_handlers(app):
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_server_error(e):
        logger.error(f"Internal server error: {str(e)}")
        return render_template('500.html'), 500
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        if isinstance(e, HTTPException):
            return e
        
        logger.error(f"Unhandled exception: {str(e)}")
        return render_template('500.html'), 500

# Register routes
def register_routes(app, limiter):
    # Home page
    @app.route('/')
    def index():
        return render_template('index.html')
    
    # Resume upload and processing
    @app.route('/upload', methods=['POST'])
    def upload():
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('index'))
        
        # Check file extension
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a PDF or TXT file.', 'error')
            return redirect(url_for('index'))
        
        # Get job domain
        job_domain = request.form.get('job_domain')
        if not job_domain:
            flash('Please select a job domain', 'error')
            return redirect(url_for('index'))
        
        file_path = None  # ensure defined for finally block
        try:
            # Save file with secure filename
            filename = secure_filename(file.filename)
            file_id = str(uuid.uuid4())
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
            file.save(file_path)
            
            # Extract text from file
            if filename.lower().endswith('.pdf'):
                resume_text = extract_text_from_pdf(file_path)
            else:  # TXT file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    resume_text = f.read()
            
            # Process resume text
            processed_text = preprocess_text(resume_text)
            
            # Extract top keywords
            top_keywords = extract_keywords(processed_text)
            
            # Store in database if user is logged in
            if 'user_id' in session:
                user_id = session['user_id']
            else:
                user_id = None
            
            # Create resume record
            resume = Resume(
                user_id=user_id,
                filename=f"{file_id}_{filename}",
                original_filename=filename,
                file_path=file_path,
                file_size=os.path.getsize(file_path),
                file_type=filename.split('.')[-1].lower(),
                extracted_text=resume_text,
                processed_text=processed_text,
                keywords=top_keywords
            )
            
            # Create job search record
            job_search = JobSearch(
                resume=resume,
                job_domain=job_domain
            )
            
            # Get job recommendations
            recommendations = get_job_recommendations(processed_text, job_domain)
            
            # Store recommendations in database
            for rec in recommendations:
                job_rec = JobRecommendation(
                    job_search=job_search,
                    company=rec['Company'],
                    role=rec['Role'],
                    location=rec['Location'],
                    date_posted=rec['Date Posted'],
                    application_link=rec['Application/Link'],
                    relevance_score=rec.get('relevance_score', 90.0) / 100.0
                )
                db.session.add(job_rec)
            
            db.session.add(resume)
            db.session.add(job_search)
            db.session.commit()
            
            # Store data in session for results page
            session['resume_id'] = resume.id
            session['job_search_id'] = job_search.id
            session['filename'] = filename
            session['job_domain'] = job_domain
            session['top_keywords'] = top_keywords[:TOP_KEYWORDS]
            
            # Redirect to results page
            return redirect(url_for('results'))
        
        except Exception as e:
            logger.error(f"Error processing resume: {str(e)}")
            flash('Error processing resume. Please try again.', 'error')
            return redirect(url_for('index'))
        finally:
            # Securely delete the file after processing
            if file_path:
                secure_delete_file(file_path)
    
    # Results page
    @app.route('/results')
    def results():
        # Check if we have results in session
        if 'job_search_id' not in session:
            flash('No results found. Please upload a resume.', 'error')
            return redirect(url_for('index'))
        
        job_search_id = session['job_search_id']
        
        # Get recommendations from database
        recommendations = JobRecommendation.query.filter_by(job_search_id=job_search_id).all()
        
        # Convert to list of dictionaries
        recommendations_list = []
        for rec in recommendations:
            recommendations_list.append({
                'Company': rec.company,
                'Role': rec.role,
                'Location': rec.location,
                'Date Posted': rec.date_posted,
                'Application/Link': rec.application_link,
                'relevance_score': int(rec.relevance_score * 100)
            })
        
        return render_template(
            'results.html',
            filename=session.get('filename', 'Unknown'),
            job_domain=session.get('job_domain', 'Unknown'),
            top_keywords=session.get('top_keywords', []),
            recommendations=recommendations_list
        )
    
    # Download results as CSV
    @app.route('/download_csv')
    def download_csv():
        # Check if we have results in session
        if 'job_search_id' not in session:
            flash('No results found. Please upload a resume.', 'error')
            return redirect(url_for('index'))
        
        job_search_id = session['job_search_id']
        
        # Get recommendations from database
        recommendations = JobRecommendation.query.filter_by(job_search_id=job_search_id).all()
        
        # Convert to list of dictionaries
        recommendations_list = []
        for rec in recommendations:
            recommendations_list.append({
                'Company': rec.company,
                'Role': rec.role,
                'Location': rec.location,
                'Date Posted': rec.date_posted,
                'Application Link': rec.application_link,
                'Relevance Score': f"{int(rec.relevance_score * 100)}%"
            })
        
        # Create DataFrame
        df = pd.DataFrame(recommendations_list)
        
        # Create CSV file in memory
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        # Create response
        return send_file(
            csv_buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name='job_recommendations.csv'
        )
    
    # API endpoint for recommendations
    @app.route('/api/recommend', methods=['POST'])
    @limiter.limit("10 per hour")
    def api_recommend():
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a PDF or TXT file.'}), 400
        
        # Get job domain
        job_domain = request.form.get('job_domain')
        if not job_domain:
            return jsonify({'error': 'Please provide a job domain'}), 400
        
        try:
            # Save file with secure filename
            filename = secure_filename(file.filename)
            file_id = str(uuid.uuid4())
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
            file.save(file_path)
            
            # Extract text from file
            if filename.lower().endswith('.pdf'):
                resume_text = extract_text_from_pdf(file_path)
            else:  # TXT file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    resume_text = f.read()
            
            # Process resume text
            processed_text = preprocess_text(resume_text)
            
            # Extract top keywords
            top_keywords = extract_keywords(processed_text)
            
            # Get job recommendations
            recommendations = get_job_recommendations(processed_text, job_domain)
            
            # Prepare response
            response = {
                'filename': filename,
                'job_domain': job_domain,
                'top_keywords': top_keywords[:TOP_KEYWORDS],
                'recommendations': recommendations
            }
            
            return jsonify(response)
        
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            # Securely delete the file after processing
            secure_delete_file(file_path)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'txt'}

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text() or ""
                text += extracted
    except Exception as e:
        logger.error(f"Failed to read PDF {file_path}: {e}")
    return text

def secure_delete_file(file_path):
    """Securely delete a file by overwriting it before deletion"""
    if os.path.exists(file_path):
        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Overwrite with random data
            with open(file_path, 'wb') as f:
                f.write(os.urandom(file_size))
            
            # Delete the file
            os.remove(file_path)
            logger.info(f"File securely deleted: {file_path}")
        except Exception as e:
            logger.error(f"Error securely deleting file: {str(e)}")

def remove_non_alpha(text):
    return re.sub(r'[^a-zA-Z\s]', ' ', text)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    tagged_words = pos_tag(word_tokens)
    
    lemmatized_words = []
    for word, tag in tagged_words:
        if tag.startswith('N'):
            lemmatized_words.append(lemmatizer.lemmatize(word, pos='n'))
        elif tag.startswith('V'):
            lemmatized_words.append(lemmatizer.lemmatize(word, pos='v'))
        elif tag.startswith('J'):
            lemmatized_words.append(lemmatizer.lemmatize(word, pos='a'))
        elif tag.startswith('R'):
            lemmatized_words.append(lemmatizer.lemmatize(word, pos='r'))
        else:
            lemmatized_words.append(word)
    
    return ' '.join(lemmatized_words)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters
    text = remove_non_alpha(text)
    
    # Lemmatize text
    text = lemmatize_text(text)
    
    # Remove stopwords
    text = remove_stopwords(text)
    
    return text

def extract_keywords(text, top_n=TOP_KEYWORDS):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Get part-of-speech tags
    tagged_tokens = pos_tag(tokens)
    
    # Filter for nouns and verbs
    important_words = [word for word, tag in tagged_tokens if tag in NOUN_TAGS or tag in VERB_TAGS]
    
    # Count word frequencies
    word_freq = {}
    for word in important_words:
        if len(word) > 2:  # Ignore very short words
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N keywords
    return [word for word, freq in sorted_words[:top_n]]

def get_job_recommendations(resume_text, job_domain, num_recommendations=NUM_POSTINGS):
    # Import here to avoid circular imports
    from WebCrawler.web_crawler import generate_sample_data
    
    # Generate sample data based on job domain
    job_data = generate_sample_data(job_domain=job_domain, num_jobs=50)
    
    # Create DataFrame
    job_df = pd.DataFrame(job_data)
    
    # Preprocess job descriptions
    job_df['processed_text'] = job_df['Role'].apply(preprocess_text)
    
    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform job descriptions
    tfidf_matrix = tfidf_vectorizer.fit_transform(job_df['processed_text'])
    
    # Transform resume text
    resume_vector = tfidf_vectorizer.transform([resume_text])
    
    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(resume_vector, tfidf_matrix).flatten()
    
    # Add similarity scores to DataFrame
    job_df['similarity'] = cosine_similarities
    
    # Sort by similarity
    job_df = job_df.sort_values('similarity', ascending=False)
    
    # Get top recommendations
    top_recommendations = job_df.head(num_recommendations)
    
    # Convert to list of dictionaries
    recommendations = []
    for _, row in top_recommendations.iterrows():
        recommendations.append({
            'Company': row['Company'],
            'Role': row['Role'],
            'Location': row['Location'],
            'Date Posted': row['Date Posted'],
            'Application/Link': row['Application/Link'],
            'relevance_score': int(row['similarity'] * 100)
        })
    
    return recommendations

# Create app instance
app = create_app()

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))