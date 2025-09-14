import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24).hex())
    DEBUG = False
    TESTING = False
    
    # SQLAlchemy configuration
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File upload configuration
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    ALLOWED_EXTENSIONS = {'pdf', 'txt'}
    
    # API configuration
    API_RATE_LIMIT = '100 per day, 10 per hour'
    
    # Job recommendation configuration
    NUM_RECOMMENDATIONS = 10
    TOP_KEYWORDS = 10
    
    # Job domain configuration
    JOB_DOMAINS = [
        'software_engineering',
        'data_science',
        'web_development',
        'finance',
        'marketing',
        'healthcare',
        'education',
        'ecommerce',
        'startup'
    ]
    
    # Security configuration
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Logging configuration
    LOG_LEVEL = 'INFO'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL', 'sqlite:///dev.db')
    SESSION_COOKIE_SECURE = False


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL', 'sqlite:///test.db')
    SESSION_COOKIE_SECURE = False


class ProductionConfig(Config):
    """Production configuration"""
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    
    # If DATABASE_URL starts with postgres://, replace it with postgresql://
    # This is needed for SQLAlchemy 1.4.x compatibility with Heroku
    if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith('postgres://'):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace('postgres://', 'postgresql://', 1)
    
    LOG_LEVEL = 'ERROR'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}