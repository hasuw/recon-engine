"""Routes for the Flask application"""

import os
import uuid
import logging
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
try:
    from flask_login import login_required, current_user  # optional
except Exception:  # pragma: no cover
    login_required = lambda f: f
    class _Anon:
        is_authenticated = False
    current_user = _Anon()

from models import db, User, Resume, JobSearch, JobRecommendation, JobPortal
from WebCrawler.web_crawler import main as crawl_jobs
from intership_recommender import process_resume, get_recommendations, generate_csv

# Configure logger
logger = logging.getLogger(__name__)

# Create Blueprint
main = Blueprint('main', __name__)
api = Blueprint('api', __name__, url_prefix='/api')

# Helper functions
def allowed_file(filename):
    """Check if the file extension is allowed"""
    ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_resume(file):
    """Save resume file and return the file path"""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Generate unique filename to prevent overwriting
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        upload_folder = current_app.config['UPLOAD_FOLDER']
        
        # Create upload folder if it doesn't exist
        os.makedirs(upload_folder, exist_ok=True)
        
        file_path = os.path.join(upload_folder, unique_filename)
        file.save(file_path)
        return file_path, unique_filename
    return None, None

def cleanup_temp_files(file_path):
    """Delete temporary files after processing"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting temporary file {file_path}: {e}")

# Routes
@main.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@main.route('/recommend', methods=['POST'])
def recommend():
    """Process resume and show job recommendations"""
    if 'resume' not in request.files:
        flash('No resume file uploaded', 'error')
        return redirect(url_for('main.index'))
    
    file = request.files['resume']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('main.index'))
    
    job_domain = request.form.get('job_domain', 'software_engineering')
    
    try:
        # Save resume
        file_path, filename = save_resume(file)
        if not file_path:
            flash('Invalid file type. Please upload PDF, TXT, or DOCX files.', 'error')
            return redirect(url_for('main.index'))
        
        # Process resume
        resume_text, skills = process_resume(file_path)
        
        # Get job recommendations
        job_data = crawl_jobs(job_domain=job_domain, min_jobs=10)
        recommendations = get_recommendations(resume_text, job_data, top_n=10)
        
        # Store in database if user is logged in
        if current_user.is_authenticated:
            # Create resume record
            resume = Resume(
                user_id=current_user.id,
                filename=filename,
                original_filename=file.filename,
                upload_date=datetime.now(),
                extracted_skills=','.join(skills)
            )
            db.session.add(resume)
            db.session.flush()  # Get resume ID without committing
            
            # Create job search record
            job_search = JobSearch(
                user_id=current_user.id,
                resume_id=resume.id,
                job_domain=job_domain,
                search_date=datetime.now()
            )
            db.session.add(job_search)
            db.session.flush()  # Get job search ID without committing
            
            # Create job recommendation records
            for job in recommendations:
                recommendation = JobRecommendation(
                    job_search_id=job_search.id,
                    job_title=job['Role'],
                    company=job['Company'],
                    location=job['Location'],
                    relevance_score=job['RelevanceScore'],
                    job_link=job['Link'],
                    recommendation_date=datetime.now()
                )
                db.session.add(recommendation)
            
            db.session.commit()
        
        # Clean up temporary files
        cleanup_temp_files(file_path)
        
        # Render results template
        return render_template(
            'results.html',
            recommendations=recommendations,
            resume_text=resume_text,
            skills=skills,
            job_domain=job_domain
        )
    
    except Exception as e:
        logger.error(f"Error processing resume: {e}")
        flash('Error processing resume. Please try again.', 'error')
        return redirect(url_for('main.index'))

@main.route('/download_csv')
def download_csv():
    """Download job recommendations as CSV"""
    # This would typically use the user's session or a temporary file
    # For now, we'll implement a placeholder
    try:
        # Generate CSV file path
        csv_file = os.path.join(current_app.config['TEMP_FOLDER'], f"recommendations_{uuid.uuid4().hex}.csv")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        
        # Write CSV data
        with open(csv_file, 'w') as f:
            f.write("Role,Company,Location,Date Posted,Relevance Score,Link\n")
            # This would normally use data from the session or database
            # For now, we'll just create a placeholder file
            
        return send_file(csv_file, as_attachment=True, download_name="job_recommendations.csv")
    
    except Exception as e:
        logger.error(f"Error generating CSV: {e}")
        flash('Error generating CSV file', 'error')
        return redirect(url_for('main.index'))

# API Routes
@api.route('/recommend', methods=['POST'])
def api_recommend():
    """API endpoint for job recommendations"""
    # Check if request has the file part
    if 'resume' not in request.files:
        return jsonify({
            'error': 'No resume file uploaded',
            'status': 'error'
        }), 400
    
    file = request.files['resume']
    if file.filename == '':
        return jsonify({
            'error': 'No file selected',
            'status': 'error'
        }), 400
    
    job_domain = request.form.get('job_domain', 'software_engineering')
    
    try:
        # Save resume
        file_path, filename = save_resume(file)
        if not file_path:
            return jsonify({
                'error': 'Invalid file type. Please upload PDF, TXT, or DOCX files.',
                'status': 'error'
            }), 400
        
        # Process resume
        resume_text, skills = process_resume(file_path)
        
        # Get job recommendations
        job_data = crawl_jobs(job_domain=job_domain, min_jobs=10)
        recommendations = get_recommendations(resume_text, job_data, top_n=10)
        
        # Clean up temporary files
        cleanup_temp_files(file_path)
        
        # Return JSON response
        return jsonify({
            'status': 'success',
            'data': {
                'recommendations': recommendations,
                'skills': skills,
                'job_domain': job_domain
            }
        })
    
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@api.route('/job_domains', methods=['GET'])
def api_job_domains():
    """API endpoint to get available job domains"""
    job_domains = current_app.config.get('JOB_DOMAINS', [
        'software_engineering', 'data_science', 'web_development', 'finance',
        'marketing', 'healthcare', 'education', 'ecommerce', 'startup'
    ])
    formatted_domains = [{
        'value': domain,
        'label': domain.replace('_', ' ').title()
    } for domain in job_domains]
    return jsonify({'status': 'success', 'data': {'domains': formatted_domains}})

@api.route('/portals', methods=['GET'])
def api_portals():
    """API endpoint to get available job portals"""
    try:
        portals = JobPortal.query.filter_by(active=True).all()
        portal_list = [{
            'id': portal.id,
            'name': portal.name,
            'url': portal.url
        } for portal in portals]
        return jsonify({'status': 'success', 'data': {'portals': portal_list}})
    except Exception as e:
        logger.error(f"API Error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500