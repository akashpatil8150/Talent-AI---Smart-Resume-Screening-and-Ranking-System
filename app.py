import os
import io
import math
import re
import uuid
import json
import traceback
import pathlib
from pathlib import Path
from datetime import datetime
import PyPDF2
import nltk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request, session, send_file, jsonify
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

# Load environment variables from .env file (for local development)
from dotenv import load_dotenv
load_dotenv()

# Import BERT components
import logging
from bert_matcher import (
    BERTEncoder,
    SimilarityCalculator,
    HybridScorer,
    EmbeddingCache,
    BERTConfig,
    BERT_AVAILABLE,
    initialize_bert_system,
    get_bert_encoder,
    get_embedding_cache
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------- Path Utility Functions -------------------------

def get_platform_path(path_str: str) -> Path:
    """Return a pathlib.Path for the given path string.

    Using pathlib ensures cross-platform compatibility between Windows
    (backslash separators) and Linux/macOS (forward-slash separators).

    Args:
        path_str: A file or directory path as a string.

    Returns:
        A pathlib.Path object representing the path.
    """
    return Path(path_str)


def ensure_directory_exists(path) -> Path:
    """Create a directory (and any missing parents) if it does not already exist.

    Equivalent to ``mkdir -p`` on Unix.  Safe to call even when the directory
    already exists (``exist_ok=True``).

    Args:
        path: A path string or pathlib.Path pointing to the directory to create.

    Returns:
        The resolved pathlib.Path of the directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


# ------------------------- Config Class -------------------------
class Config:
    """Centralized configuration management for production deployment"""
    
    # Flask Configuration
    SECRET_KEY: str = os.environ.get('SECRET_KEY', None) or os.urandom(24).hex()
    FLASK_DEBUG: bool = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    PORT: int = int(os.environ.get('PORT', 5000))
    
    # Session Cookie Security
    SESSION_COOKIE_SECURE: bool = not FLASK_DEBUG  # HTTPS only in production
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = 'Lax'
    
    # Upload Configuration
    MAX_CONTENT_LENGTH: int = 200 * 1024 * 1024  # 200MB
    
    # Data Configuration
    CANDIDATE_CSV: str = os.environ.get('CANDIDATE_CSV', 'candidates.csv')
    TOP_N_DEFAULT: int = int(os.environ.get('TOP_N_DEFAULT', '100'))
    
    # TF-IDF Configuration
    MAX_FEATURES: int = int(os.environ.get('MAX_FEATURES', '150000'))
    MIN_DF: int = int(os.environ.get('MIN_DF', '2'))
    
    # BERT Configuration
    BERT_MODEL_NAME: str = os.environ.get('BERT_MODEL_NAME', 
                                          'sentence-transformers/all-MiniLM-L6-v2')
    BERT_BATCH_SIZE: int = int(os.environ.get('BERT_BATCH_SIZE', '8'))
    BERT_DEVICE: str = os.environ.get('BERT_DEVICE', 'auto')
    BERT_FORCE_CPU: bool = os.environ.get('BERT_FORCE_CPU', 'true').lower() == 'true'
    SKIP_BERT_PRECOMPUTE: bool = os.environ.get('SKIP_BERT_PRECOMPUTE', 
                                                  'true').lower() == 'true'
    
    # Matching Configuration
    MATCHING_MODE: str = os.environ.get('MATCHING_MODE', 'hybrid')  # bert, tfidf, hybrid
    
    # Cache Configuration
    CACHE_DIR: str = os.environ.get('CACHE_DIR', '.bert_cache')
    TRANSFORMERS_CACHE: str = os.environ.get('TRANSFORMERS_CACHE', './.cache')
    
    @classmethod
    def validate(cls):
        """Validate configuration integrity"""
        errors = []
        
        # Validate SECRET_KEY length (at least 16 characters for security)
        if len(cls.SECRET_KEY) < 16:
            errors.append("SECRET_KEY must be at least 16 characters")
        
        # Validate matching mode
        if cls.MATCHING_MODE not in ['bert', 'tfidf', 'hybrid']:
            errors.append(f"Invalid MATCHING_MODE: {cls.MATCHING_MODE}. Must be 'bert', 'tfidf', or 'hybrid'")
        
        # Validate file paths
        if not get_platform_path(cls.CANDIDATE_CSV).exists():
            errors.append(f"Candidate CSV not found: {cls.CANDIDATE_CSV}")
        
        # Validate numeric ranges
        if cls.PORT < 1 or cls.PORT > 65535:
            errors.append(f"Invalid PORT: {cls.PORT}. Must be between 1-65535")
        
        if cls.BERT_BATCH_SIZE < 1:
            errors.append(f"Invalid BERT_BATCH_SIZE: {cls.BERT_BATCH_SIZE}. Must be positive")
        
        if cls.MAX_FEATURES < 1:
            errors.append(f"Invalid MAX_FEATURES: {cls.MAX_FEATURES}. Must be positive")
        
        if cls.MIN_DF < 1:
            errors.append(f"Invalid MIN_DF: {cls.MIN_DF}. Must be positive")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
    
    @classmethod
    def log_config(cls):
        """Log startup configuration (excluding secrets)"""
        logger.info("=" * 60)
        logger.info("Application Configuration:")
        logger.info("-" * 60)
        logger.info(f"  Flask Debug Mode: {cls.FLASK_DEBUG}")
        logger.info(f"  Port: {cls.PORT}")
        logger.info(f"  Max Content Length: {cls.MAX_CONTENT_LENGTH / (1024*1024):.0f}MB")
        logger.info(f"  Session Cookie Secure: {cls.SESSION_COOKIE_SECURE}")
        logger.info(f"  Session Cookie HttpOnly: {cls.SESSION_COOKIE_HTTPONLY}")
        logger.info(f"  Session Cookie SameSite: {cls.SESSION_COOKIE_SAMESITE}")
        logger.info("-" * 60)
        logger.info(f"  Candidate CSV: {cls.CANDIDATE_CSV}")
        logger.info(f"  Top N Default: {cls.TOP_N_DEFAULT}")
        logger.info("-" * 60)
        logger.info(f"  TF-IDF Max Features: {cls.MAX_FEATURES}")
        logger.info(f"  TF-IDF Min DF: {cls.MIN_DF}")
        logger.info("-" * 60)
        logger.info(f"  BERT Model: {cls.BERT_MODEL_NAME}")
        logger.info(f"  BERT Batch Size: {cls.BERT_BATCH_SIZE}")
        logger.info(f"  BERT Device: {cls.BERT_DEVICE}")
        logger.info(f"  BERT Force CPU: {cls.BERT_FORCE_CPU}")
        logger.info(f"  Skip BERT Precompute: {cls.SKIP_BERT_PRECOMPUTE}")
        logger.info("-" * 60)
        logger.info(f"  Matching Mode: {cls.MATCHING_MODE}")
        logger.info(f"  Cache Directory: {cls.CACHE_DIR}")
        logger.info(f"  Transformers Cache: {cls.TRANSFORMERS_CACHE}")
        logger.info("=" * 60)

# Legacy variable names for backward compatibility
CSV_PATH = Config.CANDIDATE_CSV
TOP_N_DEFAULT = Config.TOP_N_DEFAULT
MAX_FEATURES = Config.MAX_FEATURES
MIN_DF = Config.MIN_DF
BERT_MODEL_NAME = Config.BERT_MODEL_NAME
BERT_BATCH_SIZE = Config.BERT_BATCH_SIZE
BERT_DEVICE = Config.BERT_DEVICE
MATCHING_MODE = Config.MATCHING_MODE
MAX_PDF_SIZE = Config.MAX_CONTENT_LENGTH

# Download NLTK data only if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    # Newer NLTK versions require 'punkt_tab' alongside 'punkt'
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except Exception:
        pass
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ------------------------- Core Functions -------------------------
def clean_text(s: str) -> str:
    """Clean and normalize text"""
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^A-Za-z0-9\s+.#-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def validate_and_clean_df(df: pd.DataFrame):
    """Validate and clean the input dataframe"""
    report = {
        "total_rows": int(len(df)),
        "missing_columns": [],
        "invalid_email_count": 0,
        "empty_skills_count": 0,
        "invalid_experience_count": 0,
        "coerced_experience_to_float": 0,
        "normalized_emails": 0,
    }

    needed = ["Candidate_ID", "Name", "Email", "Experience_Years", "Skills", "Category", "Resume_Summary"]
    for col in needed:
        if col not in df.columns:
            report["missing_columns"].append(col)
    if report["missing_columns"]:
        raise ValueError(f"Missing column(s) {report['missing_columns']} in CSV. Found: {list(df.columns)}")

    # Trim whitespace on string columns
    def _safe_strip(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip()
        return s if s != "" else pd.NA
   
    for col in ["Candidate_ID", "Name", "Email", "Skills", "Category", "Resume_Summary"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_strip)

    # Normalize emails
    if "Email" in df.columns:
        before = df["Email"].copy()
        df["Email"] = df["Email"].astype(str).str.lower().where(df["Email"].notna(), other=pd.NA)
        report["normalized_emails"] = int((before != df["Email"]).sum())
        email_regex = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$")
        report["invalid_email_count"] = int((~df["Email"].fillna("").map(lambda x: bool(email_regex.match(str(x))))).sum())

    # Experience_Years numeric coercion
    if "Experience_Years" in df.columns:
        coerced = 0
        invalid = 0
        def to_float_safe(x):
            nonlocal coerced, invalid
            try:
                if pd.isna(x) or x == "":
                    invalid += 1
                    return 0.0
                f = float(str(x).strip())
                if math.isfinite(f):
                    if str(x).strip() != str(f):
                        coerced += 1
                    return f
                invalid += 1
                return 0.0
            except Exception:
                invalid += 1
                return 0.0
        df["Experience_Years"] = df["Experience_Years"].map(to_float_safe)
        report["coerced_experience_to_float"] = int(coerced)
        report["invalid_experience_count"] = int(invalid)

    # Empty skills
    if "Skills" in df.columns:
        report["empty_skills_count"] = int(df["Skills"].isna().sum())

    return df, report

def load_data(csv_path):
    """Load and process data with optimization"""
    df = pd.read_csv(csv_path)
    df, report = validate_and_clean_df(df)
   
    # Combine text fields
    skills_clean = df["Skills"].astype(str).fillna("").apply(clean_text)
    summary_clean = df["Resume_Summary"].astype(str).fillna("").apply(clean_text)
    df["_combined_text"] = (skills_clean + " " + summary_clean).str.strip()
   
    # Experience normalization [0,1]
    scaler = MinMaxScaler(feature_range=(0,1))
    exp_vals = df["Experience_Years"].fillna(0).astype(float).clip(lower=0, upper=15).values.reshape(-1,1)
    df["_exp_score"] = scaler.fit_transform(exp_vals).ravel()
   
    return df, report

def build_vectorizer(text_series):
    """Build TF-IDF vectorizer"""
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=MIN_DF, max_features=MAX_FEATURES)
    X = vect.fit_transform(text_series.tolist())
    return vect, X

# ------------------------- PDF Processing Functions -------------------------
class PDFExtractionError(ValueError):
    """Raised when a PDF cannot be read or parsed.

    Carries a *user_message* (safe to surface in API responses) and an
    optional *detail* string that is only written to logs.
    """

    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.detail = detail or user_message


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file object.

    Args:
        pdf_file: A file-like object positioned at the start of the PDF data.

    Returns:
        Extracted text as a string (may be empty for image-only PDFs).

    Raises:
        PDFExtractionError: For any problem reading or parsing the PDF,
            including oversized files and corrupted/encrypted documents.
    """
    # Check file size first (seek to end, then reset)
    try:
        pdf_file.seek(0, 2)
        file_size = pdf_file.tell()
        pdf_file.seek(0)
    except (OSError, IOError) as e:
        logger.error("PDF size check failed: %s", e, exc_info=True)
        raise PDFExtractionError(
            "Could not read the uploaded file. Please try again.",
            detail=f"Seek error during size check: {e}",
        )

    if file_size > MAX_PDF_SIZE:
        size_mb = file_size / (1024 * 1024)
        raise PDFExtractionError(
            f"The uploaded file is too large ({size_mb:.1f} MB). "
            "Please upload a PDF smaller than 200 MB.",
            detail=f"File size {size_mb:.1f} MB exceeds MAX_PDF_SIZE limit.",
        )

    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except PyPDF2.errors.PdfReadError as e:
        logger.error("Corrupted or unreadable PDF: %s", e, exc_info=True)
        raise PDFExtractionError(
            "The uploaded PDF could not be read. "
            "It may be corrupted, password-protected, or not a valid PDF.",
            detail=f"PyPDF2 PdfReadError: {e}",
        )
    except Exception as e:
        logger.error("Unexpected error extracting PDF text: %s", e, exc_info=True)
        raise PDFExtractionError(
            "An error occurred while processing the PDF. "
            "Please ensure the file is a valid, unencrypted PDF and try again.",
            detail=f"Unexpected extraction error: {e}",
        )

def extract_insights_from_resume(text):
    """Extract insights from resume text using NLP"""
    insights = {
        'name': '',
        'email': '',
        'phone': '',
        'skills': [],
        'experience_years': 0,
        'education': [],
        'companies': [],
        'summary': ''
    }
    
    # Clean text - normalize all whitespace first
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove leading special characters
    text = re.sub(r'^[/\\\-_\s]+', '', text)
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text_lower = text.lower()
   
    # Extract name - Look at the very first line before any contact info
    # The format is usually: NAME phone email linkedin location
    first_line = lines[0] if lines else ''
    
    # Try to extract name from first line (before phone/email)
    # Split by common delimiters
    parts = re.split(r'[\d@]', first_line)
    if parts and parts[0]:
        potential_name = parts[0].strip()
        # Clean up the name - remove special chars, keep only letters and spaces
        potential_name = re.sub(r'[^A-Za-z\s]', '', potential_name)
        potential_name = re.sub(r'\s+', ' ', potential_name).strip()
        
        # Check if it looks like a name (2-5 words, each word 2+ chars)
        name_words = [w for w in potential_name.split() if len(w) >= 2]
        if 2 <= len(name_words) <= 5:
            insights['name'] = ' '.join(name_words)
    
    # Fallback: Look for name in first few lines
    if not insights['name']:
        for i, line in enumerate(lines[:10]):
            # Skip lines with contact info patterns
            if re.search(r'@|http|www|\.com|phone|email|linkedin|github|\d{10}', line.lower()):
                continue
            
            # Skip lines that are section headers
            if line.lower() in ['summary', 'experience', 'education', 'skills', 'projects']:
                continue
                
            # Clean the line
            clean_line = re.sub(r'[^A-Za-z\s]', '', line)
            clean_line = re.sub(r'\s+', ' ', clean_line).strip()
            
            # Check if it looks like a name
            words = [w for w in clean_line.split() if len(w) >= 2]
            if 2 <= len(words) <= 5:
                insights['name'] = ' '.join(words)
                break
   
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        insights['email'] = emails[0]
   
    # Extract phone
    phone_patterns = [
        r'(\d{10})',  # Simple 10 digit
        r'(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
        r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
    ]
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            phone = phones[0]
            phone = re.sub(r'[^\d+]', '', phone)
            if len(phone) >= 10:
                insights['phone'] = phone
                break
   
    # Extract skills
    skill_keywords = [
        'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js', 'express',
        'django', 'flask', 'sql', 'mysql', 'postgresql', 'mongodb', 'aws', 'azure',
        'docker', 'kubernetes', 'git', 'jenkins', 'agile', 'scrum', 'machine learning',
        'ai', 'data science', 'pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn',
        'html', 'css', 'bootstrap', 'tailwind', 'php', 'c++', 'c#', '.net', 'spring',
        'hibernate', 'junit', 'selenium', 'jira', 'confluence', 'figma', 'adobe',
        'tableau', 'power bi', 'excel', 'powerpoint', 'word', 'analytics', 'statistics',
        'r programming', 'matlab', 'spark', 'hadoop', 'kafka', 'redis', 'elasticsearch',
        'graphql', 'rest api', 'microservices', 'devops', 'ci/cd', 'linux', 'unix',
        'typescript', 'golang', 'rust', 'kotlin', 'swift', 'flutter', 'react native'
    ]
   
    found_skills = []
    for skill in skill_keywords:
        if skill in text_lower:
            found_skills.append(skill.title())
    insights['skills'] = list(set(found_skills))
   
    # Extract experience years - look for patterns in the text
    exp_patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
        r'experience[:\s]*(\d+)\+?\s*(?:years?|yrs?)',
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:in|of|as)',
    ]
   
    for pattern in exp_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            try:
                years = int(matches[0])
                if 0 <= years <= 50:
                    insights['experience_years'] = years
                    break
            except:
                continue
   
    # Extract education
    education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'school', 'b.tech', 'm.tech', 'mba', 'bca', 'mca']
    sentences = nltk.sent_tokenize(text)
    education_sentences = []
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in education_keywords):
            education_sentences.append(sentence.strip())
    insights['education'] = education_sentences[:3]
   
    # Generate summary - look for SUMMARY section
    summary_match = re.search(r'SUMMARY\s*(.{50,500}?)(?:EXPERIENCE|EDUCATION|SKILLS|$)', text, re.IGNORECASE | re.DOTALL)
    if summary_match:
        insights['summary'] = summary_match.group(1).strip()[:300]
    elif sentences:
        insights['summary'] = sentences[0][:200] + "..." if len(sentences[0]) > 200 else sentences[0]
   
    return insights

def calculate_resume_score(insights, job_description):
    """Calculate a score for the resume based on job description match"""
    score = 0
    max_score = 100
   
    # Skills match (40 points)
    if insights['skills']:
        if job_description:
            job_desc_lower = job_description.lower()
            skill_matches = 0
            for skill in insights['skills']:
                skill_lower = skill.lower()
                if skill_lower in job_desc_lower:
                    skill_matches += 1
                elif any(word in job_desc_lower for word in skill_lower.split()):
                    skill_matches += 0.5
           
            if len(insights['skills']) > 0:
                skill_score = min(40, (skill_matches / len(insights['skills'])) * 40)
                score += skill_score
        else:
            skill_count = len(insights['skills'])
            if skill_count >= 8:
                score += 40
            elif skill_count >= 5:
                score += 35
            elif skill_count >= 3:
                score += 25
            else:
                score += 15
   
    # Experience match (30 points)
    if insights['experience_years'] > 0:
        if job_description:
            job_desc_lower = job_description.lower()
            exp_keywords = ['experience', 'years', 'senior', 'junior', 'entry', 'level']
            exp_mentioned = any(keyword in job_desc_lower for keyword in exp_keywords)
           
            if exp_mentioned:
                if insights['experience_years'] >= 5:
                    score += 30
                elif insights['experience_years'] >= 3:
                    score += 25
                elif insights['experience_years'] >= 1:
                    score += 20
                else:
                    score += 10
            else:
                score += min(20, insights['experience_years'] * 2)
        else:
            if insights['experience_years'] >= 5:
                score += 30
            elif insights['experience_years'] >= 3:
                score += 25
            elif insights['experience_years'] >= 1:
                score += 20
            else:
                score += 15
   
    # Completeness (20 points)
    completeness_score = 0
    if insights['name']: completeness_score += 3
    if insights['email']: completeness_score += 3
    if insights['phone']: completeness_score += 3
    if insights['skills']: completeness_score += 4
    if insights['education']: completeness_score += 3
    if insights['summary']: completeness_score += 4
    score += completeness_score
   
    # Professional keywords (10 points)
    if job_description:
        professional_keywords = ['experience', 'skills', 'project', 'team', 'lead', 'develop', 'manage', 'analysis', 'design', 'implementation']
        job_desc_lower = job_description.lower()
        keyword_matches = sum(1 for keyword in professional_keywords if keyword in job_desc_lower)
        score += min(10, keyword_matches * 1.5)
    else:
        if insights['skills'] and insights['experience_years'] > 0:
            score += 10
   
    # Bonus for having a good mix of skills and experience
    if insights['skills'] and insights['experience_years'] > 0:
        score += 5
   
    # Ensure minimum score for uploaded resumes
    if not job_description and insights['skills']:
        score = max(score, 20)
   
    return min(max_score, int(score))

# ------------------------- Global Cache -------------------------
_cached_data = None
_bert_initialized = False
_effective_matching_mode = MATCHING_MODE  # Tracks actual mode after BERT init (may fall back to tfidf)

def load_cached_data():
    """Load and cache data to avoid reloading.

    Raises:
        FileNotFoundError: When candidates.csv is missing from the configured path.
        ValueError: When the CSV is present but fails validation.
        Exception: For any other unexpected loading error.
    """
    global _cached_data
    if _cached_data is None:
        csv_path = get_platform_path(CSV_PATH)
        if not csv_path.exists():
            msg = (
                f"Candidate dataset not found at '{CSV_PATH}'. "
                "Ensure candidates.csv is present in the application directory "
                "or set the CANDIDATE_CSV environment variable to the correct path."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)
        logger.info(f"Loading candidate dataset from '{CSV_PATH}'...")
        try:
            df, report = load_data(CSV_PATH)
        except ValueError as ve:
            logger.error(f"Candidate dataset validation failed: {ve}", exc_info=True)
            raise
        except Exception as exc:
            logger.error(f"Unexpected error loading candidate dataset: {exc}", exc_info=True)
            raise
        logger.info(f"Candidate dataset loaded: {len(df)} rows.")
        try:
            vect, X = build_vectorizer(df["_combined_text"])
        except Exception as exc:
            logger.error(f"Failed to build TF-IDF vectorizer: {exc}", exc_info=True)
            raise
        _cached_data = (df, vect, X, report)
    return _cached_data

# Module-level flag set when data loading fails at startup so routes can
# return a meaningful 503 without attempting to reload on every request.
_data_load_error: str | None = None

def get_cached_data():
    """Return cached data tuple, or (None, None, None, None) on failure.

    Logs a clear error message the first time loading fails so operators can
    diagnose the problem from application logs.
    """
    global _data_load_error
    try:
        return load_cached_data()
    except FileNotFoundError as e:
        _data_load_error = str(e)
        logger.error(f"Data unavailable — candidates.csv missing: {e}")
        return None, None, None, None
    except Exception as e:
        _data_load_error = str(e)
        logger.error(f"Data unavailable — failed to load candidate dataset: {e}", exc_info=True)
        return None, None, None, None

def initialize_bert():
    """Initialize BERT system at application startup with comprehensive error handling.
    
    On failure, logs a warning and updates _effective_matching_mode to 'tfidf' so the
    application continues to operate in TF-IDF-only mode (graceful fallback).
    
    Cache regeneration logic (Requirement 5.3, 7.3):
    - Checks if the embedding cache file exists on disk at startup
    - If missing and SKIP_BERT_PRECOMPUTE=false, computes embeddings and saves to disk
    - Creates the cache directory if it doesn't exist (using pathlib)
    """
    global _bert_initialized, _effective_matching_mode
    
    if _bert_initialized:
        logger.info("BERT already initialized")
        return True
    
    if not BERT_AVAILABLE:
        logger.warning("BERT dependencies not available. Using TF-IDF only mode.")
        logger.info("To enable BERT: pip install sentence-transformers torch")
        _effective_matching_mode = "tfidf"
        logger.info(f"Matching mode updated to: {_effective_matching_mode}")
        return False
    
    try:
        logger.info("=" * 60)
        logger.info("Starting BERT initialization...")
        logger.info("=" * 60)
        
        # Create BERT config
        config = BERTConfig(
            model_name=BERT_MODEL_NAME,
            batch_size=BERT_BATCH_SIZE,
            device=BERT_DEVICE
        )
        
        # Initialize BERT system
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Device: {config.device}")
        
        success = initialize_bert_system(config)
        
        if success:
            # Pre-compute candidate embeddings (optional - can be skipped for faster startup)
            logger.info("Loading candidate data for pre-computation...")
            df, _, _, _ = get_cached_data()
            
            if df is not None:
                encoder = get_bert_encoder()
                cache = get_embedding_cache()
                
                if encoder and cache:
                    # Check if we should skip pre-computation for faster startup
                    skip_precompute = os.environ.get("SKIP_BERT_PRECOMPUTE", "false").lower() == "true"
                    
                    if skip_precompute:
                        logger.info("Skipping BERT pre-computation (SKIP_BERT_PRECOMPUTE=true)")
                        logger.info("Embeddings will be computed on-demand during queries")
                        logger.info("BERT system initialized successfully")
                        logger.info("=" * 60)
                        _bert_initialized = True
                        return True
                    
                    # --- Cache regeneration logic (Task 6.3) ---
                    # Ensure the cache directory exists (create with parents if needed)
                    cache_dir_path = ensure_directory_exists(get_platform_path(Config.CACHE_DIR))
                    logger.info(f"Cache directory ensured: {cache_dir_path}")
                    
                    # Check whether the cache file already exists on disk
                    cache_file_path = cache_dir_path / "embeddings.npz"
                    cache_exists = cache_file_path.exists()
                    
                    if cache_exists and len(cache.cache) > 0:
                        # Cache was already loaded from disk by initialize_bert_system()
                        logger.info(
                            f"Embedding cache found on disk ({cache_file_path}). "
                            f"Loaded {len(cache.cache)} embeddings — skipping recomputation."
                        )
                        logger.info("BERT system fully initialized and ready")
                        logger.info("=" * 60)
                        _bert_initialized = True
                        return True
                    
                    if not cache_exists:
                        logger.info(
                            f"No embedding cache found at {cache_file_path}. "
                            "Computing embeddings for all candidates..."
                        )
                    else:
                        logger.info(
                            f"Cache file exists at {cache_file_path} but was not loaded "
                            "(possibly empty or corrupt). Recomputing embeddings..."
                        )
                    
                    logger.info(f"Pre-computing BERT embeddings for {len(df)} candidates...")
                    logger.info("This may take 3-5 minutes. Set SKIP_BERT_PRECOMPUTE=true to skip.")
                    try:
                        # precompute_all() encodes all candidates and calls save_to_disk()
                        # which writes embeddings.npz and metadata.json to the cache directory
                        cache.precompute_all(df, encoder, text_column="_combined_text")
                        logger.info(
                            f"Embeddings saved to disk at {cache_file_path} "
                            f"({len(cache.cache)} candidates)."
                        )
                        logger.info("BERT system fully initialized and ready")
                        logger.info("=" * 60)
                        _bert_initialized = True
                        return True
                    except KeyboardInterrupt:
                        logger.warning("Pre-computation interrupted by user")
                        logger.info("BERT initialized but pre-computation incomplete. Will encode on-demand.")
                        logger.info("=" * 60)
                        _bert_initialized = True
                        return True
                    except Exception as precomp_error:
                        logger.error(f"Pre-computation failed: {precomp_error}", exc_info=True)
                        logger.warning("BERT initialized but pre-computation failed. Will encode on-demand.")
                        logger.info("=" * 60)
                        _bert_initialized = True
                        return True
                else:
                    logger.error("BERT encoder or cache not available after initialization")
            else:
                logger.error("Failed to load candidate data")
        
        # BERT initialization failed — fall back to TF-IDF
        logger.warning("BERT initialization failed. Falling back to TF-IDF mode.")
        _effective_matching_mode = "tfidf"
        logger.info(f"Matching mode updated to: {_effective_matching_mode}")
        logger.info("=" * 60)
        return False
        
    except Exception as e:
        # Unexpected error during BERT init — fall back to TF-IDF so the app stays available
        logger.error(f"BERT initialization error: {e}", exc_info=True)
        logger.warning("Falling back to TF-IDF only mode due to BERT initialization error.")
        _effective_matching_mode = "tfidf"
        logger.info(f"Matching mode updated to: {_effective_matching_mode}")
        logger.info("=" * 60)
        return False

# ------------------------- Category Filter -------------------------
def filter_by_category(jd: str, df: pd.DataFrame):
    """Filter candidates by category based on job description"""
    jd_lower = jd.lower()
    possible_categories = []

    if "data analyst" in jd_lower or "business analyst" in jd_lower:
        possible_categories.extend(["Business Analysis", "Data Science"])
    if "software" in jd_lower or "developer" in jd_lower or "engineer" in jd_lower:
        possible_categories.append("Software Engineering")
    if "product" in jd_lower:
        possible_categories.append("Product Management")
    if "hr" in jd_lower or "human resource" in jd_lower:
        possible_categories.append("HR")
    if "cloud" in jd_lower or "devops" in jd_lower:
        possible_categories.append("Cloud & DevOps")

    if possible_categories:
        return df[df["Category"].isin(possible_categories)]
    return df

def get_categories(df: pd.DataFrame):
    """Get unique categories from dataframe"""
    if df is None or "Category" not in df.columns:
        return []
    cat_series = df["Category"].dropna()
    cat_series = cat_series[cat_series.astype(str).str.strip().ne("")]
    return sorted(cat_series.astype(str).unique().tolist())

# ------------------------- Candidate Ranking -------------------------
def rank_candidates(
    job_desc: str,
    top_n: int,
    wt_text: float,
    wt_exp: float,
    min_exp: str,
    max_exp: str,
    include_skills: str,
    exclude_skills: str,
    selected_categories: list[str],
    uploaded_resume: dict | None,
    matching_mode: str = None,  # NEW: "bert", "tfidf", or "hybrid"
):
    """Rank candidates based on job description and filters"""
    # Determine matching mode — use _effective_matching_mode (which may have been downgraded
    # to 'tfidf' if BERT failed to initialize) rather than the raw config value.
    if matching_mode is None:
        matching_mode = _effective_matching_mode
    
    matching_mode = matching_mode.lower()
    logger.info(f"Ranking candidates with mode: {matching_mode}")
    
    # Get cached data
    df, vect, X, _ = get_cached_data()
    if df is None:
        logger.error("Failed to get cached data")
        return pd.DataFrame()

    def parse_num(x, default):
        try:
            x = float(x)
            if np.isfinite(x):
                return x
            return default
        except Exception:
            return default

    # Apply filters
    filtered_df = filter_by_category(job_desc, df)
    if selected_categories:
        filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]
    if filtered_df.empty:
        filtered_df = df

    min_exp_v = parse_num(min_exp, None) if min_exp else None
    max_exp_v = parse_num(max_exp, None) if max_exp else None
    if min_exp_v is not None:
        filtered_df = filtered_df[filtered_df["Experience_Years"] >= min_exp_v]
    if max_exp_v is not None:
        filtered_df = filtered_df[filtered_df["Experience_Years"] <= max_exp_v]

    def tokenize(sk: str):
        return [t.strip().lower() for t in sk.split(",") if t.strip()]

    include_tokens = tokenize(include_skills)
    exclude_tokens = tokenize(exclude_skills)
    if include_tokens:
        patt_all = [re.compile(re.escape(tok), re.IGNORECASE) for tok in include_tokens]
        mask_inc = filtered_df["Skills"].fillna("").apply(lambda s: all(p.search(s) for p in patt_all))
        filtered_df = filtered_df[mask_inc]
    if exclude_tokens:
        patt_any = [re.compile(re.escape(tok), re.IGNORECASE) for tok in exclude_tokens]
        mask_exc = filtered_df["Skills"].fillna("").apply(lambda s: any(p.search(s) for p in patt_any))
        filtered_df = filtered_df[~mask_exc]

    # Clean job description
    jd_clean = clean_text(job_desc)
    
    # Initialize similarity scores
    tfidf_sims = None
    bert_sims = None
    
    # Compute TF-IDF similarity (if needed)
    if matching_mode in ["tfidf", "hybrid"]:
        jd_vec = vect.transform([jd_clean])
        tfidf_sims = (X @ jd_vec.T).toarray().ravel()
        mask = df.index.isin(filtered_df.index)
        tfidf_sims = tfidf_sims[mask]
    
    # Compute BERT similarity (if needed and available)
    if matching_mode in ["bert", "hybrid"] and _bert_initialized:
        try:
            logger.debug(f"Computing BERT similarity for {len(filtered_df)} candidates")
            encoder = get_bert_encoder()
            cache = get_embedding_cache()
            
            if encoder and cache:
                # Encode job description
                logger.debug("Encoding job description with BERT...")
                jd_embedding = encoder.encode(jd_clean, normalize=True)
                logger.debug(f"Job description encoded: shape={jd_embedding.shape}")
                
                # Get candidate embeddings from cache
                candidate_ids = filtered_df["Candidate_ID"].tolist()
                logger.debug(f"Retrieving {len(candidate_ids)} candidate embeddings from cache...")
                
                try:
                    candidate_embeddings = cache.get_all_embeddings(candidate_ids)
                    logger.debug(f"Retrieved embeddings: shape={candidate_embeddings.shape}")
                except ValueError as cache_error:
                    # Embeddings not in cache - encode on-demand
                    logger.warning(f"Embeddings not in cache: {cache_error}")
                    logger.info("Encoding candidates on-demand (this may take a moment)...")
                    candidate_texts = filtered_df["_combined_text"].tolist()
                    candidate_embeddings = encoder.encode(candidate_texts, batch_size=8, normalize=True)
                    logger.info(f"Encoded {len(candidate_embeddings)} candidates on-demand")
                
                # Compute similarities
                logger.debug("Computing batch cosine similarities...")
                bert_sims = SimilarityCalculator.batch_cosine_similarity(
                    candidate_embeddings, 
                    jd_embedding
                )
                logger.debug(f"BERT similarities computed: min={bert_sims.min():.4f}, max={bert_sims.max():.4f}")
            else:
                logger.warning("BERT encoder or cache not available, falling back to TF-IDF")
                matching_mode = "tfidf"
                if tfidf_sims is None:
                    jd_vec = vect.transform([jd_clean])
                    tfidf_sims = (X @ jd_vec.T).toarray().ravel()
                    mask = df.index.isin(filtered_df.index)
                    tfidf_sims = tfidf_sims[mask]
        except ValueError as ve:
            # Handle missing embeddings gracefully
            logger.error(f"BERT similarity computation failed (missing embeddings): {ve}")
            logger.warning("Falling back to TF-IDF mode due to missing embeddings")
            matching_mode = "tfidf"
            if tfidf_sims is None:
                jd_vec = vect.transform([jd_clean])
                tfidf_sims = (X @ jd_vec.T).toarray().ravel()
                mask = df.index.isin(filtered_df.index)
                tfidf_sims = tfidf_sims[mask]
        except Exception as e:
            logger.error(f"BERT similarity computation failed: {e}", exc_info=True)
            logger.warning("Falling back to TF-IDF mode")
            matching_mode = "tfidf"
            if tfidf_sims is None:
                jd_vec = vect.transform([jd_clean])
                tfidf_sims = (X @ jd_vec.T).toarray().ravel()
                mask = df.index.isin(filtered_df.index)
                tfidf_sims = tfidf_sims[mask]
    elif matching_mode in ["bert", "hybrid"] and not _bert_initialized:
        logger.warning(f"BERT not initialized, falling back to TF-IDF (requested mode: {matching_mode})")
        matching_mode = "tfidf"
        if tfidf_sims is None:
            jd_vec = vect.transform([jd_clean])
            tfidf_sims = (X @ jd_vec.T).toarray().ravel()
            mask = df.index.isin(filtered_df.index)
            tfidf_sims = tfidf_sims[mask]
    
    # Combine similarities based on mode
    filtered_df = filtered_df.copy()
    
    if matching_mode == "bert" and bert_sims is not None:
        logger.info("Using BERT-only similarity scores")
        text_sims = bert_sims
    elif matching_mode == "hybrid" and bert_sims is not None and tfidf_sims is not None:
        logger.info("Using hybrid (BERT + TF-IDF) similarity scores")
        text_sims = (bert_sims + tfidf_sims) / 2.0
    else:  # tfidf mode or fallback
        logger.info("Using TF-IDF-only similarity scores")
        text_sims = tfidf_sims
    
    logger.debug(f"Text similarities: min={text_sims.min():.4f}, max={text_sims.max():.4f}, mean={text_sims.mean():.4f}")
    
    # Compute final scores
    final = wt_text * np.array(text_sims) + wt_exp * filtered_df["_exp_score"].values
    order = np.argsort(-final)
    take = order[:top_n]

    out = filtered_df.iloc[take].copy()
    logger.info(f"Ranked {len(out)} candidates (requested top {top_n})")
    
    # Add score columns based on mode
    if matching_mode == "hybrid" and bert_sims is not None and tfidf_sims is not None:
        out["BERT_Similarity"] = np.round(bert_sims[take], 4)
        out["TF-IDF_Similarity"] = np.round(tfidf_sims[take], 4)
        out["Text_Similarity"] = np.round(text_sims[take], 4)  # Combined
    elif matching_mode == "bert" and bert_sims is not None:
        out["BERT_Similarity"] = np.round(bert_sims[take], 4)
        out["Text_Similarity"] = np.round(text_sims[take], 4)
    else:  # tfidf mode
        out["Text_Similarity"] = np.round(text_sims[take], 4)
    
    out["Experience_Score"] = np.round(filtered_df["_exp_score"].values[take], 4)
    out["Final_Match_Score"] = np.round(final[take], 4)

    # Handle uploaded resume
    if uploaded_resume:
        logger.info("Processing uploaded resume")
        insights = uploaded_resume.get("insights", {})
        resume_score = insights.get("resume_score", 0)
        exp_score = min(1.0, insights.get("experience_years", 0) / 15)

        uploaded_text = uploaded_resume.get("text", "")
        uploaded_text_clean = clean_text(uploaded_text)
        
        # Compute uploaded resume similarity
        uploaded_tfidf_sim = None
        uploaded_bert_sim = None
        
        if matching_mode in ["tfidf", "hybrid"]:
            logger.debug("Computing TF-IDF similarity for uploaded resume")
            uploaded_vec = vect.transform([uploaded_text_clean])
            jd_vec = vect.transform([jd_clean])
            uploaded_tfidf_sim = (uploaded_vec @ jd_vec.T).toarray().ravel()[0]
            logger.debug(f"Uploaded resume TF-IDF similarity: {uploaded_tfidf_sim:.4f}")
        
        if matching_mode in ["bert", "hybrid"] and _bert_initialized:
            try:
                logger.debug("Computing BERT similarity for uploaded resume")
                encoder = get_bert_encoder()
                if encoder:
                    uploaded_embedding = encoder.encode(uploaded_text_clean, normalize=True)
                    jd_embedding = encoder.encode(jd_clean, normalize=True)
                    uploaded_bert_sim = SimilarityCalculator.cosine_similarity(
                        uploaded_embedding,
                        jd_embedding
                    )
                    logger.debug(f"Uploaded resume BERT similarity: {uploaded_bert_sim:.4f}")
            except Exception as e:
                logger.error(f"BERT encoding failed for uploaded resume: {e}", exc_info=True)
        
        # Determine uploaded similarity based on mode
        if matching_mode == "bert" and uploaded_bert_sim is not None:
            uploaded_similarity = uploaded_bert_sim
        elif matching_mode == "hybrid" and uploaded_bert_sim is not None and uploaded_tfidf_sim is not None:
            uploaded_similarity = (uploaded_bert_sim + uploaded_tfidf_sim) / 2.0
        elif uploaded_tfidf_sim is not None:
            uploaded_similarity = uploaded_tfidf_sim
        else:
            uploaded_similarity = resume_score / 100

        uploaded_row_data = {
            "Candidate_ID": "UPLOADED",
            "Name": insights.get("name", f"Uploaded Resume ({uploaded_resume.get('filename', 'resume.pdf')})"),
            "Email": insights.get("email", ""),
            "Experience_Years": insights.get("experience_years", 0),
            "Skills": ", ".join(insights.get("skills", [])),
            "Category": "Uploaded",
            "Resume_Summary": insights.get("summary", ""),
            "Experience_Score": exp_score,
            "Final_Match_Score": (wt_text * uploaded_similarity) + (wt_exp * exp_score),
        }
        
        # Add similarity columns based on mode
        if matching_mode == "hybrid" and uploaded_bert_sim is not None and uploaded_tfidf_sim is not None:
            uploaded_row_data["BERT_Similarity"] = uploaded_bert_sim
            uploaded_row_data["TF-IDF_Similarity"] = uploaded_tfidf_sim
            uploaded_row_data["Text_Similarity"] = uploaded_similarity
        elif matching_mode == "bert" and uploaded_bert_sim is not None:
            uploaded_row_data["BERT_Similarity"] = uploaded_bert_sim
            uploaded_row_data["Text_Similarity"] = uploaded_similarity
        else:
            uploaded_row_data["Text_Similarity"] = uploaded_similarity
        
        uploaded_row = pd.DataFrame([uploaded_row_data])

        out = pd.concat([uploaded_row, out], ignore_index=True)
        out = out.sort_values("Final_Match_Score", ascending=False).head(top_n + 1)

    # Add Rank column (1-based) after final sorting
    out = out.reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))

    # Define column order based on mode
    base_cols = [
        "Rank",
        "Candidate_ID",
        "Name",
        "Email",
        "Experience_Years",
        "Skills",
        "Category",
        "Final_Match_Score",
    ]
    
    if matching_mode == "hybrid" and "BERT_Similarity" in out.columns:
        base_cols.extend(["BERT_Similarity", "TF-IDF_Similarity", "Text_Similarity"])
    elif matching_mode == "bert" and "BERT_Similarity" in out.columns:
        base_cols.extend(["BERT_Similarity", "Text_Similarity"])
    else:
        base_cols.append("Text_Similarity")
    
    base_cols.append("Experience_Score")
    
    present_cols = [c for c in base_cols if c in out.columns]
    return out[present_cols]

# ------------------------- Flask App -------------------------
app = Flask(__name__)

# Apply configuration from Config class
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
app.config['SESSION_COOKIE_SECURE'] = Config.SESSION_COOKIE_SECURE
app.config['SESSION_COOKIE_HTTPONLY'] = Config.SESSION_COOKIE_HTTPONLY
app.config['SESSION_COOKIE_SAMESITE'] = Config.SESSION_COOKIE_SAMESITE

# Set secret_key for backward compatibility
app.secret_key = Config.SECRET_KEY


@app.after_request
def add_security_headers(response):
    """Add security headers to every HTTP response (Requirement 15.4).

    Headers applied:
    - X-Content-Type-Options: prevents MIME-type sniffing attacks.
    - X-Frame-Options: blocks the app from being embedded in iframes (clickjacking).
    - Content-Security-Policy: restricts resource origins to reduce XSS attack surface.
    - X-XSS-Protection: enables the browser's built-in XSS filter (legacy browsers).
    - Referrer-Policy: limits referrer information sent with cross-origin requests.
    """
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:;"
    )
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle HTTP 413 Request Entity Too Large errors.

    Flask raises this automatically when an uploaded file exceeds
    app.config['MAX_CONTENT_LENGTH'] (currently 200 MB).
    """
    max_mb = Config.MAX_CONTENT_LENGTH // (1024 * 1024)
    logger.warning("HTTP 413: Upload rejected — file exceeds %d MB limit.", max_mb)
    return jsonify({
        'error': 'File too large',
        'message': (
            f'The uploaded file exceeds the maximum allowed size of {max_mb} MB. '
            'Please upload a smaller file and try again.'
        )
    }), 413


@app.errorhandler(MemoryError)
def handle_memory_error(exc):
    """Handle MemoryError — raised when the process runs out of available RAM.

    Logs the event with resource_type="memory" and the configured memory limit
    so operators can diagnose OOM conditions in production (Requirement 10.6).

    Returns HTTP 503 Service Unavailable so load-balancers / clients know the
    server is temporarily unable to fulfil the request.
    """
    memory_limit_mb = Config.MAX_CONTENT_LENGTH // (1024 * 1024)
    logger.error(
        "Resource limit exceeded: resource_type=memory, limit=%dMB — %s",
        memory_limit_mb,
        exc,
        exc_info=True,
    )
    return jsonify({
        'error': 'Service temporarily unavailable',
        'message': (
            'The server ran out of memory while processing your request. '
            'Try reducing the batch size or uploading a smaller file.'
        )
    }), 503


@app.errorhandler(TimeoutError)
def handle_timeout_error(exc):
    """Handle TimeoutError — raised when an operation exceeds its time limit.

    Logs the event with resource_type="timeout" and the gunicorn worker timeout
    value so operators can identify slow operations (Requirement 10.6).

    Returns HTTP 504 Gateway Timeout to signal that the upstream operation did
    not complete within the allowed time window.
    """
    # Gunicorn default timeout is 120 s (configured in Procfile / render.yaml).
    timeout_seconds = int(os.environ.get("GUNICORN_TIMEOUT", 120))
    logger.error(
        "Resource limit exceeded: resource_type=timeout, limit=%ds — %s",
        timeout_seconds,
        exc,
        exc_info=True,
    )
    return jsonify({
        'error': 'Request timed out',
        'message': (
            'The request took too long to process and was cancelled. '
            'Try a smaller dataset or a simpler query.'
        )
    }), 504


@app.errorhandler(OSError)
def handle_os_error(exc):
    """Handle OSError — specifically catches ENOSPC (disk full) conditions.

    When errno is ENOSPC the handler logs resource_type="disk" together with
    the cache directory path so operators know which volume is full
    (Requirement 10.6).  All other OSError variants are re-raised so the
    global Exception handler can process them normally.

    Returns HTTP 507 Insufficient Storage for disk-full errors.
    """
    import errno as errno_module
    if exc.errno == errno_module.ENOSPC:
        cache_dir = Config.CACHE_DIR
        logger.error(
            "Resource limit exceeded: resource_type=disk, path=%s — no space left on device: %s",
            cache_dir,
            exc,
            exc_info=True,
        )
        return jsonify({
            'error': 'Insufficient storage',
            'message': (
                'The server disk is full and cannot complete the request. '
                'Please contact the administrator to free up disk space.'
            )
        }), 507
    # Not a disk-full error — let the global handler deal with it.
    return handle_unexpected_exception(exc)


@app.errorhandler(Exception)
def handle_unexpected_exception(exc):
    """Catch-all handler for any unhandled exception (Requirements 12.5, 12.6).

    Before falling through to the generic 500 response, checks for resource
    limit conditions that may arrive as plain Exception subclasses rather than
    the specific types registered above (e.g. MemoryError raised inside a
    C-extension, or an OSError with ENOSPC from a library call).

    Logs the full exception with stack trace so operators can diagnose the
    problem from application logs, but returns only a generic message to the
    caller so that internal details (file paths, stack traces, configuration)
    are never exposed to users.

    Args:
        exc: The unhandled exception that propagated to Flask.

    Returns:
        A JSON response with an appropriate HTTP status code and a safe message.
    """
    import errno as errno_module

    # --- Resource limit checks (Requirement 10.6) ---
    if isinstance(exc, MemoryError):
        return handle_memory_error(exc)

    if isinstance(exc, TimeoutError):
        return handle_timeout_error(exc)

    if isinstance(exc, OSError) and exc.errno == errno_module.ENOSPC:
        return handle_os_error(exc)

    # --- Generic unhandled exception ---
    logger.error(
        "Unhandled exception: %s",
        exc,
        exc_info=True,
    )
    return jsonify({
        'error': 'An unexpected error occurred. Please try again later.'
    }), 500


# Validate and log configuration at startup (runs under both gunicorn and __main__)
logger.info("Starting AI Resume Screening Application...")
try:
    Config.validate()
    Config.log_config()
except ValueError as e:
    logger.error(f"Configuration validation failed: {e}")
    logger.error("Please check your environment variables and try again.")

# Global variable to store last ranking results (avoids session size limits)
_last_ranking_results = None

@app.route('/')
def index():
    df, _, _, _ = get_cached_data()
    if df is None:
        logger.error("Index route: candidate data not available — returning 503")
        return jsonify({
            'error': 'Service temporarily unavailable',
            'message': (
                'The candidate dataset could not be loaded. '
                'Please check that candidates.csv exists and try again later.'
            )
        }), 503
    
    categories = get_categories(df)
    sample_count = len(df)
    
    return render_template('index.html', 
                         categories=categories,
                         sample_count=sample_count,
                         top_n_default=TOP_N_DEFAULT)

@app.route('/rank', methods=['POST'])
def rank():
    try:
        job_desc = request.form.get('job_description', '').strip()
        if not job_desc:
            return jsonify({'error': 'Please enter a Job Description'}), 400
        
        top_n = int(request.form.get('top_n', TOP_N_DEFAULT))
        wt_text = float(request.form.get('wt_text', 0.85))
        wt_exp = float(request.form.get('wt_exp', 0.15))
        min_exp = request.form.get('min_exp', '')
        max_exp = request.form.get('max_exp', '')
        include_skills = request.form.get('include_skills', '')
        exclude_skills = request.form.get('exclude_skills', '')
        selected_categories = request.form.getlist('selected_categories')
        matching_mode = request.form.get('matching_mode', _effective_matching_mode)  # NEW
        
        uploaded_resume_data = session.get('uploaded_resume_data')
        
        results = rank_candidates(
            job_desc=job_desc,
            top_n=top_n,
            wt_text=wt_text,
            wt_exp=wt_exp,
            min_exp=min_exp,
            max_exp=max_exp,
            include_skills=include_skills,
            exclude_skills=exclude_skills,
            selected_categories=selected_categories,
            uploaded_resume=uploaded_resume_data,
            matching_mode=matching_mode,  # NEW
        )
        
        # Store results in global variable (avoids session size limits)
        global _last_ranking_results
        _last_ranking_results = results
        
        return jsonify({
            'success': True,
            'html': results.to_html(classes='table table-dark table-striped table-hover', index=False, border=0)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['resume']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are allowed. Please upload a .pdf file.'}), 400

        # Read file bytes once so we can pass a BytesIO to the extractor
        # (avoids double-seek issues with Werkzeug FileStorage objects).
        raw_bytes = file.read()
        buf = io.BytesIO(raw_bytes)

        # --- PDF extraction (may raise PDFExtractionError) ---
        try:
            text = extract_text_from_pdf(buf)
        except PDFExtractionError as pdf_err:
            # Log the full detail (including stack trace) for operators.
            logger.error(
                "PDF extraction failed for file '%s': %s",
                secure_filename(file.filename),
                pdf_err.detail,
                exc_info=True,
            )
            # Return a user-friendly message without internal details.
            return jsonify({'error': pdf_err.user_message}), 400

        # Store raw text for debugging
        session['raw_resume_text'] = text[:1000]  # Store first 1000 chars

        insights = extract_insights_from_resume(text)

        if not insights.get("summary") and text:
            compact = " ".join([ln.strip() for ln in text.splitlines() if ln.strip()])
            insights["summary"] = compact[:300]

        job_desc = request.form.get('job_description', '')
        if job_desc:
            insights["resume_score"] = calculate_resume_score(insights, job_desc)
        else:
            insights["resume_score"] = 0

        session['uploaded_resume_data'] = {
            "insights": insights,
            "text": text,
            "filename": secure_filename(file.filename),
        }

        return jsonify({
            'success': True,
            'insights': {
                'name': insights.get('name') or 'Not found',
                'email': insights.get('email') or '',
                'experience_years': insights.get('experience_years', 0),
                'skills_count': len(insights.get('skills', [])),
                'resume_score': insights.get('resume_score', 0)
            },
            'debug_text': text[:500]  # Return first 500 chars for debugging
        })
    except Exception as e:
        # Catch-all for unexpected errors in this route (e.g. session write
        # failures, insight extraction crashes).  Log with full stack trace
        # but return a generic message so internal details are never exposed.
        logger.error(
            "Unexpected error in /upload_resume: %s", e, exc_info=True
        )
        return jsonify({
            'error': 'An unexpected error occurred while processing your resume. Please try again.'
        }), 500

@app.route('/download_csv')
def download_csv():
    try:
        global _last_ranking_results
        
        if _last_ranking_results is None or _last_ranking_results.empty:
            return "No ranking results available. Please rank candidates first.", 400
        
        csv_bytes = _last_ranking_results.to_csv(index=False).encode('utf-8')
        
        return send_file(
            io.BytesIO(csv_bytes),
            mimetype='text/csv',
            as_attachment=True,
            download_name='ranked_candidates.csv'
        )
    except Exception as e:
        logger.error(f"Download CSV error: {e}", exc_info=True)
        return str(e), 500

@app.route('/analysis')
def analysis():
    df, _, _, _ = get_cached_data()
    if df is None:
        logger.error("Analysis route: candidate data not available — returning 503")
        return jsonify({
            'error': 'Service temporarily unavailable',
            'message': (
                'The candidate dataset could not be loaded. '
                'Please check that candidates.csv exists and try again later.'
            )
        }), 503
    
    charts = {}
    
    # 1. Category Distribution
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x="Category", data=df, order=df["Category"].value_counts().index, ax=ax, palette="viridis")
        plt.xticks(rotation=45, ha='right')
        plt.title("Candidate Categories Distribution")
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        charts['category_dist'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    except:
        charts['category_dist'] = None
    
    # 2. Experience Distribution
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df["Experience_Years"], bins=20, kde=True, ax=ax, color="#60a5fa")
        plt.title("Experience Distribution")
        plt.xlabel("Years of Experience")
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        charts['exp_dist'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    except:
        charts['exp_dist'] = None
    
    # 3. Experience by Category
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x="Category", y="Experience_Years", data=df, ax=ax, palette="Set2")
        plt.xticks(rotation=45, ha='right')
        plt.title("Experience by Category")
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        charts['exp_by_cat'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    except:
        charts['exp_by_cat'] = None
    
    # 4. Top Skills
    try:
        skills_series = df.get("Skills", pd.Series(dtype=str)).fillna("").astype(str)
        tokens = []
        for row in skills_series:
            tokens.extend([t.strip().lower() for t in row.split(",") if t.strip()])
        if tokens:
            vc = pd.Series(tokens).value_counts().head(20)
            fig, ax = plt.subplots(figsize=(10, 8))
            vc.sort_values().plot(kind="barh", ax=ax, color="#60a5fa")
            ax.set_xlabel("Count")
            ax.set_ylabel("Skill")
            plt.title("Top 20 Skills")
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            charts['top_skills'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
    except:
        charts['top_skills'] = None
    
    # 5. Email Domains
    try:
        domains = (
            df.get("Email", pd.Series(dtype=str))
              .dropna().astype(str).str.extract(r"@(.+)")[0].str.lower()
        )
        vc = domains.value_counts().head(10)
        if not vc.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            vc.sort_values().plot(kind="barh", ax=ax, color="#34d399")
            ax.set_xlabel("Count")
            ax.set_ylabel("Domain")
            plt.title("Top 10 Email Domains")
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            charts['email_domains'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
    except:
        charts['email_domains'] = None
    
    # 6. Category Share Pie
    try:
        vc = df["Category"].fillna("Unknown").value_counts()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        plt.title("Category Share")
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        charts['category_pie'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
    except:
        charts['category_pie'] = None
    
    return render_template('analysis.html', 
                         charts=charts,
                         total_rows=len(df),
                         total_cols=len(df.columns))

@app.route('/health')
def health():
    """Health check endpoint for deployment platforms.
    
    Returns HTTP 200 with status='healthy' when the application is ready,
    or HTTP 503 with status='unhealthy' when critical components are unavailable.
    """
    try:
        checks = {}

        # --- data_loaded: candidates.csv loaded into memory ---
        df, _, _, _ = get_cached_data()
        data_loaded = df is not None and len(df) > 0
        checks['data_loaded'] = data_loaded

        # --- BERT availability and initialization status ---
        checks['bert_available'] = bool(BERT_AVAILABLE)
        checks['bert_initialized'] = bool(_bert_initialized)

        # --- Embedding cache status ---
        cache = get_embedding_cache() if BERT_AVAILABLE else None
        if cache is not None:
            checks['cache_available'] = True
            try:
                # Count cached embeddings if the cache exposes that information
                cached_count = len(cache) if hasattr(cache, '__len__') else (
                    len(cache.embeddings) if hasattr(cache, 'embeddings') and cache.embeddings is not None else 0
                )
            except Exception:
                cached_count = 0
            checks['cached_embeddings'] = cached_count
        else:
            checks['cache_available'] = False
            checks['cached_embeddings'] = 0

        # --- Current matching mode (reflects actual capability after BERT init) ---
        checks['matching_mode'] = _effective_matching_mode

        # Determine overall health: data must be loaded for the app to be useful
        is_healthy = data_loaded
        status_code = 200 if is_healthy else 503
        overall_status = 'healthy' if is_healthy else 'unhealthy'

        response_body = {
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'checks': checks,
        }

        return jsonify(response_body), status_code

    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'error': str(e),
        }), 503


if __name__ == '__main__':
    # Log startup configuration (debug mode, port, matching mode)
    logger.info("=" * 60)
    logger.info("Starting AI Resume Screening Application (dev server)")
    logger.info(f"  Debug mode : {Config.FLASK_DEBUG}")
    logger.info(f"  Port       : {Config.PORT}")
    logger.info(f"  Matching   : {Config.MATCHING_MODE}")
    logger.info(f"  Host       : 0.0.0.0")
    logger.info("=" * 60)

    # Pre-load candidate data at startup so the first request is fast.
    # A missing or corrupt candidates.csv is logged clearly; the app still
    # starts so the /health endpoint can report the unhealthy state.
    logger.info("Loading candidate dataset at startup...")
    try:
        load_cached_data()
        logger.info("Candidate dataset loaded successfully.")
    except FileNotFoundError as e:
        logger.error(
            "STARTUP ERROR — candidates.csv not found. "
            f"Detail: {e}. "
            "The application will start but all ranking routes will return HTTP 503 "
            "until the dataset is available."
        )
    except Exception as e:
        logger.error(
            f"STARTUP ERROR — failed to load candidate dataset: {e}. "
            "The application will start but all ranking routes will return HTTP 503.",
            exc_info=True,
        )

    # Initialize BERT system at startup
    logger.info("Initializing BERT system...")
    initialize_bert()
    logger.info("Application ready")

    # Start Flask application — bind to 0.0.0.0 for external connections.
    # DEBUG and PORT are read from environment variables via the Config class.
    app.run(debug=Config.FLASK_DEBUG, host='0.0.0.0', port=Config.PORT)
