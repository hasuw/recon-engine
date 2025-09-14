import logging
from app import create_app
from models import db, init_db as models_init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_db():
    """Initialize the database with tables and seed data"""
    app = create_app('development')
    with app.app_context():
        # Initialize database with default data from models.py (creates tables and seeds)
        models_init_db(app)
        logger.info("Database initialized successfully!")
# Legacy seeding removed; models.init_db handles creating default domains and portals.

if __name__ == "__main__":
    try:
        init_db()
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise