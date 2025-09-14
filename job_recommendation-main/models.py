import os
import logging
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON
import uuid

# Initialize SQLAlchemy instance
db = SQLAlchemy()

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class User(db.Model):
    """User model for storing user information"""
    __tablename__ = 'users'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(120), unique=True, nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    resumes = db.relationship('Resume', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.id}'


class Resume(db.Model):
    """Resume model for storing resume metadata"""
    __tablename__ = 'resumes'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=True)  # Path if stored on disk
    file_size = db.Column(db.Integer, nullable=False)
    file_type = db.Column(db.String(50), nullable=False)  # PDF, TXT, etc.
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Extracted data
    extracted_text = db.Column(db.Text, nullable=True)  # Raw text from resume
    processed_text = db.Column(db.Text, nullable=True)  # Processed text (after NLP)
    keywords = db.Column(JSON, nullable=True)  # Top keywords extracted
    
    # Relationships
    job_searches = db.relationship('JobSearch', backref='resume', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Resume {self.id} - {self.original_filename}>'


class JobSearch(db.Model):
    """Job search model for storing search parameters and results"""
    __tablename__ = 'job_searches'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    resume_id = db.Column(db.String(36), db.ForeignKey('resumes.id'), nullable=False)
    job_domain = db.Column(db.String(100), nullable=False)  # e.g., software_engineering, data_science
    search_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    recommendations = db.relationship('JobRecommendation', backref='job_search', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<JobSearch {self.id} - {self.job_domain}>'


class JobRecommendation(db.Model):
    """Job recommendation model for storing recommended jobs"""
    __tablename__ = 'job_recommendations'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_search_id = db.Column(db.String(36), db.ForeignKey('job_searches.id'), nullable=False)
    company = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(255), nullable=False)
    date_posted = db.Column(db.String(100), nullable=True)  # Keep as string for flexibility
    application_link = db.Column(db.String(1024), nullable=True)
    salary = db.Column(db.String(100), nullable=True)  # Salary range as string
    experience = db.Column(db.String(100), nullable=True)  # Experience requirements
    job_domain_id = db.Column(db.String(36), db.ForeignKey('job_domains.id'), nullable=True)
    relevance_score = db.Column(db.Float, nullable=False)  # Cosine similarity score
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    job_domain = db.relationship('JobDomain', backref='recommendations', lazy=True)
    
    def __repr__(self):
        return f'<JobRecommendation {self.id} - {self.role} at {self.company}>'


class JobDomain(db.Model):
    """Job domain model for categorizing jobs"""
    __tablename__ = 'job_domains'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    job_portals = db.relationship('JobPortal', backref='domain', lazy=True)
    
    def __repr__(self):
        return f'<JobDomain {self.name}>'


class JobPortal(db.Model):
    """Job portal model for storing job portal configurations"""
    __tablename__ = 'job_portals'

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False, unique=True)
    url = db.Column(db.String(512), nullable=False)
    domain_id = db.Column(db.String(36), db.ForeignKey('job_domains.id'), nullable=True)
    active = db.Column(db.Boolean, default=True)
    config = db.Column(JSON, nullable=True)  # Store scraping configuration
    last_scraped = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<JobPortal {self.name}>'


def init_db(app):
    """Initialize the database with the Flask app"""
    try:
        db.init_app(app)
        with app.app_context():
            db.create_all()
            
            # Create default job domains if they don't exist
            domains = [
                {'name': 'Software Development', 'description': 'Software engineering, web development, mobile app development'},
                {'name': 'Data Science', 'description': 'Data analysis, machine learning, AI, statistics'},
                {'name': 'Design', 'description': 'UI/UX design, graphic design, product design'},
                {'name': 'Marketing', 'description': 'Digital marketing, content marketing, SEO'},
                {'name': 'Finance', 'description': 'Accounting, financial analysis, investment banking'},
                {'name': 'Healthcare', 'description': 'Medical, pharmaceutical, healthcare administration'},
                {'name': 'Education', 'description': 'Teaching, training, educational technology'},
                {'name': 'Engineering', 'description': 'Mechanical, electrical, civil engineering'}
            ]
            
            for domain_data in domains:
                domain = JobDomain.query.filter_by(name=domain_data['name']).first()
                if not domain:
                    domain = JobDomain(id=str(uuid.uuid4()), **domain_data)
                    db.session.add(domain)
                    logger.info(f"Created job domain: {domain_data['name']}")
            
            # Create default job portals if they don't exist
            portals = [
                {'name': 'Naukri', 'url': 'https://www.naukri.com', 'active': True},
                {'name': 'LinkedIn', 'url': 'https://www.linkedin.com/jobs', 'active': True},
                {'name': 'Internshala', 'url': 'https://internshala.com', 'active': True},
                {'name': 'Indeed', 'url': 'https://www.indeed.co.in', 'active': True},
                {'name': 'Monster', 'url': 'https://www.monsterindia.com', 'active': True}
            ]
            
            for portal_data in portals:
                portal = JobPortal.query.filter_by(name=portal_data['name']).first()
                if not portal:
                    portal = JobPortal(id=str(uuid.uuid4()), **portal_data)
                    db.session.add(portal)
                    logger.info(f"Created job portal: {portal_data['name']}")
            
            db.session.commit()
            logger.info('Database initialized successfully')
    except Exception as e:
        logger.error(f'Error initializing database: {e}')
        raise