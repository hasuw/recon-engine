import unittest
import os
import tempfile
from app import app

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_home_page(self):
        """Test that the home page loads correctly"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        # Look for the main heading text in index.html
        self.assertIn(b'AI-Powered Job Recommender for India', response.data)
    
    def test_404_page(self):
        """Test that the 404 page works"""
        response = self.app.get('/nonexistent-page')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'Page Not Found', response.data)
    
    def test_api_endpoint_no_file(self):
        """Test API endpoint with no file"""
        response = self.app.post('/api/recommend')
        self.assertEqual(response.status_code, 400)
        self.assertIn('No file part', response.get_json().get('error', ''))
    
    def test_api_endpoint_empty_filename(self):
        """Test API endpoint with empty filename"""
        data = {
            'file': (tempfile.BytesIO(b''), ''),
            'job_domain': 'software_engineering'
        }
        response = self.app.post('/api/recommend', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn('No file selected', response.get_json().get('error', ''))
    
    def test_api_endpoint_invalid_file_type(self):
        """Test API endpoint with invalid file type"""
        data = {
            'file': (tempfile.BytesIO(b'test content'), 'test.jpg'),
            'job_domain': 'software_engineering'
        }
        response = self.app.post('/api/recommend', data=data)
        self.assertEqual(response.status_code, 400)
        self.assertIn('Invalid file type', response.get_json().get('error', ''))

if __name__ == '__main__':
    unittest.main()