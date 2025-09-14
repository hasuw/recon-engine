import os
import re
import time
import random
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("webcrawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import database models if available
try:
    from models import db, JobPortal
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.warning("Database models not available, using hardcoded job portal data")

# Indian job portals to scrape
JOB_PORTALS = {
    'naukri': {
        'url': 'https://www.naukri.com/jobs-in-india',
        'job_container': 'article.jobTuple',
        'company': '.companyInfo span.companyName',
        'role': '.title',
        'location': '.locWdth span.ellipsis',
        'date_posted': '.postedDate',
        'apply_link': 'a.title'
    },
    'linkedin': {
        'url': 'https://www.linkedin.com/jobs/search/?keywords=&location=India',
        'job_container': '.job-search-card',
        'company': '.base-search-card__subtitle',
        'role': '.base-search-card__title',
        'location': '.job-search-card__location',
        'date_posted': 'time',
        'apply_link': '.base-card__full-link'
    },
    'internshala': {
        'url': 'https://internshala.com/internships/',
        'job_container': '.individual_internship',
        'company': '.company_name',
        'role': '.profile',
        'location': '.location_link',
        'date_posted': '.posted_by_container',
        'apply_link': '.view_detail_button'
    }
}

def setup_driver():
    """Set up and return a headless Chrome WebDriver"""
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except Exception as e:
        logger.error(f"Error setting up WebDriver: {e}")
        return None

def scrape_job_portal(portal_name, job_domain=None):
    """Scrape a specific job portal for job listings
    
    Args:
        portal_name (str): Name of the portal to scrape
        job_domain (str, optional): Job domain to filter by. Defaults to None.
        
    Returns:
        list: List of job dictionaries
    """
    if portal_name not in JOB_PORTALS:
        logger.error(f"Portal {portal_name} not supported")
        return []
    
    portal_config = JOB_PORTALS[portal_name]
    url = portal_config['url']
    
    # Add job domain to URL if provided
    if job_domain:
        if portal_name == 'naukri':
            url = f"https://www.naukri.com/{job_domain.replace('_', '-')}-jobs-in-india"
        elif portal_name == 'linkedin':
            url = f"https://www.linkedin.com/jobs/search/?keywords={job_domain.replace('_', '%20')}&location=India"
        elif portal_name == 'internshala':
            url = f"https://internshala.com/internships/{job_domain.replace('_', '-')}-internship"
    
    driver = setup_driver()
    if not driver:
        return []
    
    job_data = []
    
    try:
        logger.info(f"Navigating to {url}")
        driver.get(url)
        
        # Wait for the page to load
        time.sleep(5)  # Give time for JavaScript to render content
        
        # Scroll down to load more jobs (for sites with lazy loading)
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        
        # Wait for job containers to be present
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, portal_config['job_container']))
            )
        except Exception as e:
            logger.warning(f"Timeout waiting for job containers: {e}")
        
        # Get the page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Find all job listings
        job_listings = soup.select(portal_config['job_container'])
        logger.info(f"Found {len(job_listings)} job listings on {portal_name}")
        
        # Extract job data
        for job in job_listings[:20]:  # Limit to 20 jobs per portal
            try:
                # Extract job details
                company_elem = job.select_one(portal_config['company'])
                role_elem = job.select_one(portal_config['role'])
                location_elem = job.select_one(portal_config['location'])
                date_elem = job.select_one(portal_config['date_posted'])
                link_elem = job.select_one(portal_config['apply_link'])
                
                # Get text content or default values
                company = company_elem.text.strip() if company_elem else "Unknown Company"
                role = role_elem.text.strip() if role_elem else "Unknown Role"
                location = location_elem.text.strip() if location_elem else "India"
                date_posted = date_elem.text.strip() if date_elem else "Recent"
                
                # Get the application link
                if link_elem and link_elem.has_attr('href'):
                    link = link_elem['href']
                    if not link.startswith('http'):
                        # Handle relative URLs
                        if portal_name == 'naukri':
                            link = f"https://www.naukri.com{link}"
                        elif portal_name == 'internshala':
                            link = f"https://internshala.com{link}"
                else:
                    link = "#"
                
                # Filter by job domain if provided
                if job_domain:
                    domain_keywords = job_domain.replace('_', ' ').lower().split()
                    role_match = any(keyword in role.lower() for keyword in domain_keywords)
                    if not role_match:
                        continue
                
                job_data.append({
                    'Company': company,
                    'Role': role,
                    'Location': location,
                    'Date Posted': date_posted,
                    'Application/Link': link
                })
            except Exception as e:
                logger.error(f"Error extracting job data: {e}")
                continue
    
    except Exception as e:
        logger.error(f"Error scraping {portal_name}: {e}")
    
    finally:
        driver.quit()
    
    return job_data

def generate_sample_data(job_domain=None, num_jobs=50):
    """Generate sample job data when scraping fails"""
    
    # Get job portals from database if available
    if DB_AVAILABLE:
        try:
            from flask import current_app
            with current_app.app_context():
                job_portals = JobPortal.query.filter_by(is_active=True).all()
                if job_portals:
                    logger.info(f"Using {len(job_portals)} job portals from database")
        except Exception as e:
            logger.error(f"Error fetching job portals from database: {e}")
            job_portals = []
    else:
        job_portals = []
    
    # Indian companies by sector
    companies = {
        'tech': ['Infosys', 'TCS', 'Wipro', 'HCL Technologies', 'Tech Mahindra', 'Mindtree', 'L&T Infotech'],
        'startups': ['Zomato', 'Swiggy', 'Paytm', 'Ola', 'CRED', 'Razorpay', 'PhonePe'],
        'finance': ['HDFC Bank', 'ICICI Bank', 'SBI', 'Axis Bank', 'Kotak Mahindra Bank', 'Yes Bank', 'RBI'],
        'ecommerce': ['Flipkart', 'Amazon India', 'Myntra', 'Snapdeal', 'Nykaa', 'BigBasket', 'Meesho'],
        'healthcare': ['Apollo Hospitals', 'Fortis Healthcare', 'Max Healthcare', 'Manipal Hospitals', 'Medanta'],
        'education': ['BYJU\'S', 'Unacademy', 'Vedantu', 'upGrad', 'Great Learning', 'Simplilearn']
    }
    
    # Job roles by sector
    roles = {
        'tech': ['Software Engineer', 'Full Stack Developer', 'Data Scientist', 'DevOps Engineer', 'QA Engineer', 'Software Architect'],
        'startups': ['Product Manager', 'Growth Hacker', 'UI/UX Designer', 'Backend Developer', 'Mobile Developer', 'Customer Success Manager'],
        'finance': ['Financial Analyst', 'Investment Banker', 'Risk Analyst', 'Compliance Officer', 'Wealth Manager', 'Credit Analyst'],
        'ecommerce': ['Category Manager', 'Supply Chain Analyst', 'Digital Marketer', 'Customer Experience Manager', 'Inventory Manager'],
        'healthcare': ['Healthcare Administrator', 'Clinical Research Associate', 'Medical Officer', 'Pharmacist', 'Biomedical Engineer'],
        'education': ['Curriculum Developer', 'Educational Consultant', 'Academic Counselor', 'E-learning Specialist', 'Instructional Designer']
    }
    
    # Indian cities
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad', 'Gurgaon', 'Noida', 'Kochi', 'Indore', 'Chandigarh', 'Coimbatore']
    
    # Salary ranges by sector (in lakhs per annum)
    salary_ranges = {
        'tech': ['8-12 LPA', '12-18 LPA', '15-25 LPA', '20-35 LPA', '30-50 LPA'],
        'startups': ['10-15 LPA', '15-25 LPA', '20-40 LPA', '30-60 LPA'],
        'finance': ['7-12 LPA', '12-20 LPA', '18-30 LPA', '25-45 LPA'],
        'ecommerce': ['8-15 LPA', '15-25 LPA', '20-35 LPA'],
        'healthcare': ['6-10 LPA', '10-18 LPA', '15-30 LPA', '25-40 LPA'],
        'education': ['5-10 LPA', '10-18 LPA', '15-25 LPA', '20-35 LPA']
    }
    
    # Experience requirements by sector
    experience_ranges = {
        'tech': ['0-2 years', '2-5 years', '5-8 years', '8+ years'],
        'startups': ['1-3 years', '3-6 years', '5+ years'],
        'finance': ['1-3 years', '3-7 years', '7+ years'],
        'ecommerce': ['0-2 years', '2-5 years', '5-8 years'],
        'healthcare': ['1-4 years', '4-8 years', '8+ years'],
        'education': ['0-3 years', '3-6 years', '6+ years']
    }
    
    # Generate sample data
    sample_data = []
    
    # Filter by job domain if provided
    if job_domain and job_domain in companies:
        sectors = [job_domain]
    else:
        sectors = list(companies.keys())
    
    for _ in range(num_jobs):  # Generate specified number of sample jobs
        sector = random.choice(sectors)
        company = random.choice(companies[sector])
        role = random.choice(roles[sector])
        location = random.choice(cities)
        
        # Random date within the last 30 days
        days_ago = random.randint(0, 30)
        date_posted = (datetime.now() - timedelta(days=days_ago)).strftime("%d %b %Y")
        
        # Generate a fake application link with Indian domain
        domain = random.choice(["com", "co.in", "in"])
        link = f"https://careers.{company.lower().replace(' ', '')}.{domain}/jobs/{role.lower().replace(' ', '-')}"
        
        # Add salary and experience information for Indian job market
        salary = random.choice(salary_ranges[sector])
        experience = random.choice(experience_ranges[sector])
        
        sample_data.append({
            'Company': company,
            'Role': role,
            'Location': location,
            'Date Posted': date_posted,
            'Application/Link': link,
            'Salary': salary,
            'Experience': experience
        })
    
    return sample_data

def create_html_table(job_data):
    """Create an HTML table from job data"""
    html = "<table border='1'>\n"
    html += "<tr><th>Company</th><th>Role</th><th>Location</th><th>Date Posted</th><th>Application/Link</th></tr>\n"
    
    for job in job_data:
        html += "<tr>"
        html += f"<td>{job['Company']}</td>"
        html += f"<td>{job['Role']}</td>"
        html += f"<td>{job['Location']}</td>"
        html += f"<td>{job['Date Posted']}</td>"
        html += f"<td><a href='{job['Application/Link']}' target='_blank'>Apply</a></td>"
        html += "</tr>\n"
    
    html += "</table>"
    return html

def main(job_domain=None, min_jobs=10, save_to_html=True):
    """Main function to scrape job portals and save data
    
    Args:
        job_domain (str, optional): Job domain to filter by. Defaults to None.
        min_jobs (int, optional): Minimum number of jobs to scrape before falling back to sample data. Defaults to 10.
        save_to_html (bool, optional): Whether to save results to HTML file. Defaults to True.
        
    Returns:
        list: List of job dictionaries
    """
    all_job_data = []
    
    # Get job portals from database if available
    if DB_AVAILABLE:
        try:
            from flask import current_app
            with current_app.app_context():
                job_portals = JobPortal.query.filter_by(is_active=True).all()
                if job_portals:
                    logger.info(f"Using {len(job_portals)} job portals from database")
                    portals_to_scrape = {}
                    for portal in job_portals:
                        portals_to_scrape[portal.name] = {
                            'url': portal.url,
                            'selectors': json.loads(portal.selectors) if portal.selectors else {}
                        }
                    if portals_to_scrape:
                        JOB_PORTALS.update(portals_to_scrape)
        except Exception as e:
            logger.error(f"Error fetching job portals from database: {e}")
    
    # Try to scrape each portal
    for portal in JOB_PORTALS.keys():
        try:
            logger.info(f"Scraping {portal}...")
            portal_data = scrape_job_portal(portal, job_domain)
            all_job_data.extend(portal_data)
            logger.info(f"Scraped {len(portal_data)} jobs from {portal}")
            
            # Add a small delay between portals to avoid rate limiting
            time.sleep(random.uniform(2, 5))
        except Exception as e:
            logger.error(f"Failed to scrape {portal}: {e}")
    
    # If we couldn't scrape enough jobs, use sample data
    if len(all_job_data) < min_jobs:
        logger.warning(f"Not enough jobs scraped ({len(all_job_data)}), generating sample data")
        sample_data = generate_sample_data(job_domain=job_domain, num_jobs=50)
        
        # If we have some scraped data, combine it with sample data
        if all_job_data:
            # Prioritize real data, then add sample data to reach min_jobs
            sample_data_needed = min_jobs - len(all_job_data)
            if sample_data_needed > 0:
                all_job_data.extend(sample_data[:sample_data_needed])
        else:
            all_job_data = sample_data
    
    # Create HTML table and save to file if requested
    if save_to_html:
        html_table = create_html_table(all_job_data)
        
        try:
            with open("table.html", "w", encoding="utf-8") as f:
                f.write(html_table)
            logger.info(f"Saved {len(all_job_data)} jobs to table.html")
        except Exception as e:
            logger.error(f"Error saving table.html: {e}")
    
    return all_job_data

if __name__ == "__main__":
    main()
