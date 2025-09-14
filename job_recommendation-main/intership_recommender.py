import os
import re
import logging
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
import docx2txt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logger
logger = logging.getLogger(__name__)

# POS TAG AND Word Lemmatizer

#{ Part-of-speech constants
ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
#}
POS_LIST = [NOUN, VERB, ADJ, ADV]

NUM_POSTING = 50

TOP_N_KEYWORDS = 5

def download_nltk_resources():
    """
    Download necessary NLTK resources for text processing.
    """
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK resources downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {e}")
        raise

def process_resume(file_path):
    """
    Process a resume file and extract text and skills.
    
    Args:
        file_path (str): Path to the resume file
        
    Returns:
        tuple: (resume_text, skills_list)
    """
    try:
        # Ensure NLTK resources are downloaded
        download_nltk_resources()
        
        # Read the resume file
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            resume_text = read_pdf(file_path)
        elif file_extension == '.docx':
            resume_text = docx2txt.process(file_path)
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                resume_text = file.read()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Pre-process the resume text
        processed_text = pre_process_resume(resume_text)
        
        # Extract skills from the resume
        skills = extract_skills(resume_text)
        
        return processed_text, skills
    
    except Exception as e:
        logger.error(f"Error processing resume: {e}")
        raise

def extract_skills(text):
    """
    Extract skills from resume text.
    
    Args:
        text (str): Resume text
        
    Returns:
        list: List of skills extracted from the resume
    """
    # Common technical skills
    technical_skills = [
        # Programming Languages
        'Python', 'Java', 'JavaScript', 'C++', 'C#', 'Ruby', 'PHP', 'Swift', 'Kotlin', 'Go', 'Rust',
        'TypeScript', 'Scala', 'Perl', 'R', 'MATLAB', 'Bash', 'Shell', 'PowerShell', 'SQL', 'NoSQL',
        
        # Web Development
        'HTML', 'CSS', 'React', 'Angular', 'Vue', 'Node.js', 'Express', 'Django', 'Flask', 'Spring',
        'ASP.NET', 'Laravel', 'Ruby on Rails', 'jQuery', 'Bootstrap', 'Tailwind', 'Material UI',
        'Redux', 'GraphQL', 'REST API', 'SOAP', 'XML', 'JSON', 'AJAX',
        
        # Data Science & ML
        'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision', 'Data Mining', 'Data Analysis',
        'Data Visualization', 'Statistical Analysis', 'Big Data', 'TensorFlow', 'PyTorch', 'Keras',
        'Scikit-learn', 'Pandas', 'NumPy', 'SciPy', 'Matplotlib', 'Seaborn', 'Tableau', 'Power BI',
        'SPSS', 'SAS', 'Hadoop', 'Spark', 'Kafka', 'Airflow',
        
        # Databases
        'MySQL', 'PostgreSQL', 'Oracle', 'SQL Server', 'MongoDB', 'Cassandra', 'Redis', 'Elasticsearch',
        'DynamoDB', 'Firebase', 'Neo4j', 'Couchbase', 'MariaDB', 'SQLite',
        
        # DevOps & Cloud
        'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'Jenkins', 'Git', 'GitHub', 'GitLab', 'Bitbucket',
        'CI/CD', 'Terraform', 'Ansible', 'Puppet', 'Chef', 'Prometheus', 'Grafana', 'ELK Stack',
        'Nginx', 'Apache', 'Linux', 'Unix', 'Windows Server',
        
        # Mobile Development
        'Android', 'iOS', 'React Native', 'Flutter', 'Xamarin', 'Ionic', 'Swift UI', 'Jetpack Compose',
        
        # Other Technical Skills
        'Agile', 'Scrum', 'Kanban', 'JIRA', 'Confluence', 'Trello', 'Slack', 'Microsoft Teams',
        'Cybersecurity', 'Blockchain', 'IoT', 'AR/VR', 'Game Development', 'Unity', 'Unreal Engine',
        
        # Indian Tech Skills
        'Infosys Nia', 'TCS Ignio', 'Wipro Holmes', 'HCL DRYiCE', 'Tech Mahindra Gaia',
        'Zoho Creator', 'Freshworks', 'Razorpay', 'Paytm API', 'PhonePe API'
    ]
    
    # Common soft skills
    soft_skills = [
        'Communication', 'Teamwork', 'Problem Solving', 'Critical Thinking', 'Creativity',
        'Leadership', 'Time Management', 'Adaptability', 'Collaboration', 'Emotional Intelligence',
        'Conflict Resolution', 'Decision Making', 'Negotiation', 'Presentation', 'Public Speaking',
        'Customer Service', 'Interpersonal Skills', 'Active Listening', 'Empathy', 'Patience',
        'Flexibility', 'Work Ethic', 'Attention to Detail', 'Organization', 'Multitasking',
        'Self-motivation', 'Analytical Skills', 'Research', 'Strategic Planning', 'Project Management'
    ]
    
    # Combine all skills
    all_skills = technical_skills + soft_skills
    
    # Extract skills from text
    found_skills = []
    text_lower = text.lower()
    
    for skill in all_skills:
        skill_pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(skill_pattern, text_lower):
            found_skills.append(skill)
    
    return found_skills

def get_recommendations(resume_text, job_data, top_n=10):
    """
    Get job recommendations based on resume text.
    
    Args:
        resume_text (str): Processed resume text
        job_data (list): List of job dictionaries
        top_n (int): Number of top recommendations to return
        
    Returns:
        list: List of top job recommendations
    """
    try:
        # Convert job data to DataFrame
        job_df = pd.DataFrame(job_data)
        
        # Ensure required columns exist
        required_columns = ['Role', 'Company', 'Location']
        for col in required_columns:
            if col not in job_df.columns:
                logger.error(f"Required column '{col}' not found in job data")
                return []
        
        # Add data column for TF-IDF processing
        job_df['data'] = job_df['Role']
        
        # Pre-process job data
        job_df = pre_process_data_job(job_df)
        
        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(job_df['data'])
        
        # Calculate TF-IDF vector for resume
        resume_vector = tfidf_vectorizer.transform([resume_text])
        
        # Calculate cosine similarities
        cosine_similarities = cosine_similarity(resume_vector, tfidf_matrix)
        
        # Get indices of jobs sorted by similarity
        job_indices = cosine_similarities.argsort()[0][::-1]
        
        # Extract top recommendations
        top_recommendations = []
        for idx in job_indices[:top_n]:
            job = job_df.iloc[idx].to_dict()
            
            # Add relevance score (scaled to 0-100)
            relevance = cosine_similarities[0][idx] * 100
            job['RelevanceScore'] = round(relevance, 1)
            
            top_recommendations.append(job)
        
        return top_recommendations
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return []
        st.write(df_resume_sorted.to_html(escape=False), unsafe_allow_html=True)


def generate_csv(job_recommendations, output_path=None):
    """
    Generate a CSV file from job recommendations.
    
    Args:
        job_recommendations (list): List of job recommendation dictionaries
        output_path (str, optional): Path to save the CSV file. If None, returns CSV content as string.
        
    Returns:
        str or None: CSV content as string if output_path is None, otherwise None
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(job_recommendations)
        
        # Reorder columns
        columns = ['Role', 'Company', 'Location', 'DatePosted', 'RelevanceScore', 'Link', 'Salary', 'Experience']
        available_columns = [col for col in columns if col in df.columns]
        
        # Add any remaining columns
        for col in df.columns:
            if col not in available_columns:
                available_columns.append(col)
        
        # Reorder DataFrame
        df = df[available_columns]
        
        if output_path:
            # Save to file
            df.to_csv(output_path, index=False)
            logger.info(f"CSV file saved to {output_path}")
            return None
        else:
            # Return as string
            return df.to_csv(index=False)
    
    except Exception as e:
        logger.error(f"Error generating CSV: {e}")
        raise


def get_top_keywords(resume_text, job_df, num_keywords=5):
    """
    Extract top keywords from resume text based on TF-IDF scores.
    
    Args:
        resume_text (str): Processed resume text
        job_df (DataFrame): Job data DataFrame
        num_keywords (int): Number of keywords to extract
        
    Returns:
        list: List of top keywords
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(job_df['data'])
    
    # Calculate the TF-IDF vector for the input word
    resume_text_vector = tfidf_vectorizer.transform([resume_text])

    # Get feature names from the TF-IDF vectorizer
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

    # Get the indices of features sorted by TF-IDF scores
    feature_indices = np.argsort(resume_text_vector.toarray()[0])[::-1]

    # Extract the top N features (keywords and skills)
    top_features = feature_names[feature_indices][:num_keywords]

    return top_features.tolist()

def format_job_data_for_display(job_recommendations):
    """
    Format job recommendations for display in HTML.
    
    Args:
        job_recommendations (list): List of job recommendation dictionaries
        
    Returns:
        list: List of job dictionaries with formatted fields for display
    """
    try:
        formatted_jobs = []
        
        for job in job_recommendations:
            formatted_job = job.copy()
            
            # Format relevance score
            if 'RelevanceScore' in formatted_job:
                formatted_job['RelevanceScore'] = f"{formatted_job['RelevanceScore']}%"
            
            # Format link as HTML
            if 'Link' in formatted_job:
                formatted_job['Link'] = f'<a href="{formatted_job["Link"]}" target="_blank">Apply</a>'
            
            formatted_jobs.append(formatted_job)
        
        return formatted_jobs
    
    except Exception as e:
        logger.error(f"Error formatting job data: {e}")
        return job_recommendations

def prepare_job_data(job_data):
    """
    Prepare job data for processing.
    
    Args:
        job_data (list): List of job dictionaries
        
    Returns:
        pd.DataFrame: DataFrame with prepared job data
    """
    try:
        # Convert to DataFrame
        job_df = pd.DataFrame(job_data)
        
        # Standardize column names
        column_mapping = {
            'Company': 'Company',
            'Role': 'Role',
            'Location': 'Location',
            'Application/Link': 'Link',
            'Date Posted': 'DatePosted'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in job_df.columns and old_col != new_col:
                job_df[new_col] = job_df[old_col]
        
        # Add data column for TF-IDF processing if not exists
        if 'data' not in job_df.columns:
            job_df['data'] = job_df['Role']
        
        # Pre-process job data
        job_df = pre_process_data_job(job_df)
        
        return job_df
    
    except Exception as e:
        logger.error(f"Error preparing job data: {e}")
        raise

def make_clickable(link):
    """
    Create a clickable HTML link.

    Parameters:
    str: link - The URL for the hyperlink.

    Returns:
    str: An HTML-formatted string containing a clickable link.
    """
    text = "Apply"
    return f'<a href="{link}" target="_blank">{text}</a>'


def read_pdf(pdf_file):
    """
    Read text content from a PDF file.

    Parameters:
    str: pdf_file - The path to the PDF file.

    Returns:
    str: Text content extracted from the PDF file.
    """
    try:
        # Open the PDF file
        reader = PdfReader(pdf_file)
        
        # Extract text from all pages
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return text
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        raise


def return_data_list():
    """
    Returns a list of job data from the HTML file.

    Returns:
    list: A list of job data.
    """
    
    job_data_list = get_job_data_list()
    return job_data_list

def get_job_data_list():
    """
    Extracts job data from an HTML file using BeautifulSoup.

    Returns:
    list: A list of job data.
    """
    
    job_data_list = []
    
    try:
        with open('job_data.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all job listings
        job_listings = soup.find_all('div', class_='job-listing')
        
        for job in job_listings:
            company = job.find('div', class_='company').text.strip()
            role = job.find('div', class_='role').text.strip()
            location = job.find('div', class_='location').text.strip()
            link = job.find('a', class_='apply-link')['href']
            date_posted = job.find('div', class_='date-posted').text.strip()
            
            job_data_list.append([company, role, location, link, date_posted])
    
    except Exception as e:
        logger.error(f"Error extracting job data: {e}")
        # Generate sample data if HTML file is not found or has issues
        job_data_list = [
            ['Google', 'Software Engineer', 'Mountain View, CA', 'https://careers.google.com', '2023-01-15'],
            ['Microsoft', 'Data Scientist', 'Redmond, WA', 'https://careers.microsoft.com', '2023-01-14'],
            ['Amazon', 'Product Manager', 'Seattle, WA', 'https://amazon.jobs', '2023-01-13'],
            ['Facebook', 'UX Designer', 'Menlo Park, CA', 'https://facebook.com/careers', '2023-01-12'],
            ['Apple', 'iOS Developer', 'Cupertino, CA', 'https://apple.com/jobs', '2023-01-11'],
            ['Netflix', 'Content Strategist', 'Los Gatos, CA', 'https://netflix.jobs', '2023-01-10'],
            ['Tesla', 'Electrical Engineer', 'Palo Alto, CA', 'https://tesla.com/careers', '2023-01-09'],
            ['Twitter', 'Backend Developer', 'San Francisco, CA', 'https://careers.twitter.com', '2023-01-08'],
            ['LinkedIn', 'Frontend Developer', 'Sunnyvale, CA', 'https://linkedin.com/jobs', '2023-01-07'],
            ['Uber', 'Machine Learning Engineer', 'San Francisco, CA', 'https://uber.com/careers', '2023-01-06']
        ]
    
    return job_data_list


def keep_alpha_char(text):
    """
    Remove non-alphabetic characters from the input text, keeping only alphabetical characters.

    Args:
        text (str): Input text containing alphanumeric and non-alphanumeric characters

    Returns:
        str: A cleaned string containing only alphabetical characters
    """
    try:
        if not isinstance(text, str):
            return ""
        alpha_only_string = re.sub(r'[^a-zA-Z]', ' ', text)
        cleaned_string = re.sub(r'\s+', ' ', alpha_only_string)
        return cleaned_string
    except Exception as e:
        logger.error(f"Error in keep_alpha_char: {e}")
        return text if isinstance(text, str) else ""



def nltk_pos_tagger(nltk_tag):
    """
    Map NLTK POS tags to WordNet POS tags.

    Args:
        nltk_tag (str): The POS tag assigned by NLTK

    Returns:
        wordnet POS tag or None if no mapping exists
    """
    try:
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None
    except Exception as e:
        logger.error(f"Error in nltk_pos_tagger: {e}")
        return None


    
def lemmatize_sentence(sentence):
    """
    Lemmatize the words in the input sentence.

    Args:
        sentence (str): Input sentence to be lemmatized

    Returns:
        str: The lemmatized sentence
    """
    try:
        if not isinstance(sentence, str) or not sentence.strip():
            return ""
            
        lemmatizer = WordNetLemmatizer()
        
        # Part-of-speech tagging using NLTK
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
        
        # Map NLTK POS tags to WordNet POS tags
        wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
        
        lemmatized_sentence = []
        
        # Lemmatize each word based on its POS tag
        for word, tag in wordnet_tagged:
            if tag is not None:      
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
            else:
                lemmatized_sentence.append(word)
        
        # Join the lemmatized words back into a sentence
        return " ".join(lemmatized_sentence)
    except Exception as e:
        logger.error(f"Error in lemmatize_sentence: {e}")
        return sentence if isinstance(sentence, str) else ""


def remove_stop_words(text):
    """
    Remove English stop words from the input text.

    Args:
        text (str): The input text from which stop words will be removed.

    Returns:
        str: The input text with English stop words removed.
    """
    try:
        if not isinstance(text, str) or not text.strip():
            return ""
            
        # Tokenize the text
        words = nltk.word_tokenize(str(text))

        # Get the list of English stop words
        stop_words = set(stopwords.words('english'))

        # Remove stop words from the list of words
        filtered_words = [word for word in words if word.lower() not in stop_words]

        # Join the filtered words back into a text
        filtered_text = ' '.join(filtered_words)

        return filtered_text
    except Exception as e:
        logger.error(f"Error in remove_stop_words: {e}")
        return text if isinstance(text, str) else ""



def recommend_job(resume_text, tfidf_matrix, tfidf_vectorizer, df, num_recommendations=10, job_domain=None):
    """
    Recommends jobs based on an input word using TF-IDF and cosine similarity.

    Args:
        resume_text (str): The input word or text for which job recommendations are sought.
        tfidf_matrix (scipy.sparse.csr_matrix): The TF-IDF matrix representing job descriptions.
        tfidf_vectorizer (TfidfVectorizer): The TF-IDF vectorizer used for vectorizing input words.
        df (pd.DataFrame): The DataFrame containing job information.
        num_recommendations (int): Number of recommendations to return.
        job_domain (str, optional): Filter jobs by domain.

    Returns:
        pd.DataFrame: A table of recommended jobs sorted by similarity to the input word.
    """
    try:
        logger.info(f"Finding job recommendations for resume with {len(resume_text)} characters")
        
        # Calculate the TF-IDF vector for the input word
        resume_text_vector = tfidf_vectorizer.transform([resume_text])

        # Calculate cosine similarities between the input word vector and job vectors
        cosine_similarities = cosine_similarity(resume_text_vector, tfidf_matrix)

        # Get indices of jobs sorted by similarity (highest to lowest)
        job_indices = cosine_similarities.argsort()[0][::-1]

        # Filter by job domain if specified
        if job_domain and isinstance(job_domain, str) and 'Role' in df.columns:
            domain_mask = df['Role'].str.contains(job_domain, case=False, na=False)
            job_indices = [idx for idx in job_indices if idx < len(domain_mask) and domain_mask.iloc[idx]]

        # Extract the jobs corresponding to the top recommendations
        top_recommendations_full = [df.iloc[index] for index in job_indices[:num_recommendations]]

        # Add relevance scores
        for i, job in enumerate(top_recommendations_full):
            job['RelevanceScore'] = round(cosine_similarities[0][job_indices[i]] * 100, 1)

        return pd.DataFrame(top_recommendations_full)
    except Exception as e:
        logger.error(f"Error in recommend_job: {e}")
        return pd.DataFrame()

def pre_process_resume(resume_text):
    """
    Preprocesses a resume text by removing stop words, lemmatizing, and keeping only alphabet characters.

    Args:
        resume_text (str): The text content of the resume.

    Returns:
        str: The preprocessed resume text.
    """
    try:
        if not isinstance(resume_text, str) or not resume_text.strip():
            logger.warning("Empty resume text provided for preprocessing")
            return ""
            
        # Remove non-alphabetic characters and keep only alphabet characters
        resume_text = keep_alpha_char(resume_text)

        # Lemmatize the words in the resume text
        resume_text = lemmatize_sentence(resume_text)

        # Remove stop words from the resume text
        resume_text = remove_stop_words(resume_text)

        # Convert the resume text to lowercase
        resume_text = resume_text.lower()

        logger.info(f"Resume preprocessing complete, resulting in {len(resume_text)} characters")
        return resume_text
    except Exception as e:
        logger.error(f"Error in pre_process_resume: {e}")
        return "" if not isinstance(resume_text, str) else resume_text.lower()


def pre_process_data_job(job_df):
    """
    Preprocess the job_df database by removing stop words, returning the words to their base form,
    and keeping only alphabet characters.

    Args:
        job_df (pd.DataFrame): The job database containing job descriptions.
          job_df["Role"] is the column that contains the title and the role of the internship.

    Returns:
        pd.DataFrame: A table of recommended jobs that has the pre-processed data for the column "Role".
    """
    try:
        if job_df.empty or 'Role' not in job_df.columns:
            logger.warning("Empty job DataFrame or missing 'Role' column")
            return job_df
            
        # Create a copy to avoid modifying the original DataFrame
        processed_df = job_df.copy()
        
        # Drop rows with missing values in the "Role" column
        processed_df.dropna(subset=['Role'], inplace=True)

        # Apply preprocessing steps directly to the DataFrame
        processed_df['data'] = processed_df['Role'].apply(keep_alpha_char)
        processed_df['data'] = processed_df['data'].apply(lemmatize_sentence)
        processed_df['data'] = processed_df['data'].apply(remove_stop_words)
        processed_df['data'] = processed_df['data'].str.lower()

        # Removing inactive job listings
        if 'Link' in processed_df.columns:
            processed_df = processed_df[~processed_df['Link'].str.contains('🔒', na=False)]

        logger.info(f"Processed {len(processed_df)} job listings")
        return processed_df
    except Exception as e:
        logger.error(f"Error in pre_process_data_job: {e}")
        return job_df


def return_table_job(resume_text, job_df, num_recommendations=10, job_domain=None):
    """
    Generates a table of recommended jobs based on the provided resume text and a job database.

    Args:
        resume_text (str): The resume text or content of the uploaded resume file.
        job_df (pd.DataFrame): The job database containing job descriptions.
        num_recommendations (int): Number of recommendations to return.
        job_domain (str, optional): Filter jobs by domain.

    Returns:
        pd.DataFrame: A table of recommended jobs sorted by relevance to the provided resume.
        The table includes columns for job titles, locations, dates, and links to job postings.
    """
    try:
        if not isinstance(resume_text, str) or not resume_text.strip():
            logger.warning("Empty resume text provided")
            return pd.DataFrame()
            
        if job_df.empty or 'data' not in job_df.columns:
            logger.warning("Empty job DataFrame or missing 'data' column")
            return pd.DataFrame()

        # Create TF-IDF vectorizer and transform job descriptions
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(job_df['data'])
        
        # Recommend jobs using cosine similarity
        recommended_jobs = recommend_job(resume_text, tfidf_matrix, tfidf_vectorizer, job_df, 
                                        num_recommendations=num_recommendations, job_domain=job_domain)
        
        # Format links as clickable HTML if Link column exists
        if 'Link' in recommended_jobs.columns:
            recommended_jobs['Link'] = recommended_jobs['Link'].apply(make_clickable)
        
        logger.info(f"Generated table with {len(recommended_jobs)} job recommendations")
        return recommended_jobs
    except Exception as e:
        logger.error(f"Error in return_table_job: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    main()
