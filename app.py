import os
import io
import math
import re
import uuid
import json
import PyPDF2
import nltk
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# ------------------------- Config -------------------------
CSV_PATH = os.environ.get("CANDIDATE_CSV", "candidates.csv")
TOP_N_DEFAULT = int(os.environ.get("TOP_N_DEFAULT", "100"))
MAX_FEATURES = int(os.environ.get("MAX_FEATURES", "150000"))
MIN_DF = int(os.environ.get("MIN_DF", "2"))

# PDF Upload Config - 200MB maximum
MAX_PDF_SIZE = 200 * 1024 * 1024  # 200MB max file size

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
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file with size validation"""
    try:
        # Check file size
        pdf_file.seek(0, 2)  # Seek to end
        file_size = pdf_file.tell()
        pdf_file.seek(0)  # Reset to beginning
       
        if file_size > MAX_PDF_SIZE:
            raise Exception(f"PDF file size ({file_size / (1024*1024):.1f}MB) exceeds maximum allowed size (200MB)")
       
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

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
   
    # Extract name
    name_patterns = [
        r'(?:name|full name|contact)[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?:\s+email|\s+phone|\s+address)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)(?:\s+Phone|\s+Email|\s+Address)',
    ]
   
    for pattern in name_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            name = matches[0].strip()
            name_parts = name.split()
            if len(name_parts) >= 2 and all(len(part) > 1 for part in name_parts):
                cleaned_parts = []
                for part in name_parts:
                    if part.lower() in ['phone', 'email', 'address', 'contact', 'mobile', 'tel']:
                        break
                    cleaned_parts.append(part)
               
                if len(cleaned_parts) >= 2:
                    insights['name'] = ' '.join(cleaned_parts)
                    break
   
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        insights['email'] = emails[0]
   
    # Extract phone
    phone_pattern = r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
    phones = re.findall(phone_pattern, text)
    if phones:
        insights['phone'] = ''.join(phones[0])
   
    # Extract skills
    skill_keywords = [
        'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js', 'express',
        'django', 'flask', 'sql', 'mysql', 'postgresql', 'mongodb', 'aws', 'azure',
        'docker', 'kubernetes', 'git', 'jenkins', 'agile', 'scrum', 'machine learning',
        'ai', 'data science', 'pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn',
        'html', 'css', 'bootstrap', 'tailwind', 'php', 'c++', 'c#', '.net', 'spring',
        'hibernate', 'junit', 'selenium', 'jira', 'confluence', 'figma', 'adobe'
    ]
   
    text_lower = text.lower()
    found_skills = []
    for skill in skill_keywords:
        if skill in text_lower:
            found_skills.append(skill.title())
    insights['skills'] = list(set(found_skills))
   
    # Extract experience years
    exp_patterns = [
        r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?experience',
        r'experience[:\s](\d+)\s(?:years?|yrs?)',
        r'(\d+)\s*(?:years?|yrs?)\s*in\s*(?:the\s*)?field',
        r'(\d+)\s*(?:years?|yrs?)\s*(?:in\s+)?(?:software|development|programming|data|analysis)',
        r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s+)?(?:work|professional|industry)',
        r'(\d+)\s*(?:years?|yrs?)\s*(?:in\s+)?(?:it|technology|tech)',
        r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s+)?(?:coding|programming)'
    ]
   
    for pattern in exp_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            try:
                insights['experience_years'] = int(matches[0])
                break
            except:
                continue
   
    # Extract education
    education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'school']
    sentences = nltk.sent_tokenize(text)
    education_sentences = []
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in education_keywords):
            education_sentences.append(sentence.strip())
    insights['education'] = education_sentences[:3]
   
    # Generate summary
    sentences = nltk.sent_tokenize(text)
    if sentences:
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
@st.cache_data
def load_cached_data():
    """Load and cache data to avoid reloading"""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    df, report = load_data(CSV_PATH)
    vect, X = build_vectorizer(df["_combined_text"])
    return df, vect, X, report

def ensure_model_loaded():
    """Ensure model is loaded and cached"""
    if 'cached_data' not in st.session_state:
        try:
            st.session_state.cached_data = load_cached_data()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.session_state.cached_data = None

def get_cached_data():
    """Get cached data"""
    ensure_model_loaded()
    if st.session_state.cached_data is None:
        return None, None, None, None
    return st.session_state.cached_data

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
):
    """Rank candidates based on job description and filters"""
    df, vect, X, _ = get_cached_data()
    if df is None:
        return pd.DataFrame()

    def parse_num(x, default):
        try:
            x = float(x)
            if np.isfinite(x):
                return x
            return default
        except Exception:
            return default

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

    jd_clean = clean_text(job_desc)
    jd_vec = vect.transform([jd_clean])
    sims = (X @ jd_vec.T).toarray().ravel()

    mask = df.index.isin(filtered_df.index)
    sims = sims[mask]
    filtered_df = filtered_df.copy()

    final = wt_text * np.array(sims) + wt_exp * filtered_df["_exp_score"].values
    order = np.argsort(-final)
    take = order[:top_n]

    out = filtered_df.iloc[take].copy()
    out["Text_Similarity"] = np.round(sims[take], 4)
    out["Experience_Score"] = np.round(filtered_df["_exp_score"].values[take], 4)
    out["Final_Match_Score"] = np.round(final[take], 4)

    if uploaded_resume:
        insights = uploaded_resume.get("insights", {})
        resume_score = insights.get("resume_score", 0)
        exp_score = min(1.0, insights.get("experience_years", 0) / 15)

        uploaded_text = uploaded_resume.get("text", "")
        if uploaded_text and job_desc:
            uploaded_text_clean = clean_text(uploaded_text)
            uploaded_vec = vect.transform([uploaded_text_clean])
            uploaded_similarity = (uploaded_vec @ jd_vec.T).toarray().ravel()[0]
        else:
            if uploaded_text:
                uploaded_text_clean = clean_text(uploaded_text)
                uploaded_vec = vect.transform([uploaded_text_clean])
                skills_text = " ".join(insights.get("skills", []))
                if skills_text:
                    skills_vec = vect.transform([skills_text])
                    uploaded_similarity = (uploaded_vec @ skills_vec.T).toarray().ravel()[0]
                else:
                    uploaded_similarity = 0.3
            else:
                uploaded_similarity = resume_score / 100

        uploaded_row = pd.DataFrame([
            {
                "Candidate_ID": "UPLOADED",
                "Name": insights.get("name", f"Uploaded Resume ({uploaded_resume.get('filename', 'resume.pdf')})"),
                "Email": insights.get("email", ""),
                "Experience_Years": insights.get("experience_years", 0),
                "Skills": ", ".join(insights.get("skills", [])),
                "Category": "Uploaded",
                "Resume_Summary": insights.get("summary", ""),
                "Text_Similarity": uploaded_similarity,
                "Experience_Score": exp_score,
                "Final_Match_Score": (wt_text * uploaded_similarity) + (wt_exp * exp_score),
            }
        ])

        out = pd.concat([uploaded_row, out], ignore_index=True)
        out = out.sort_values("Final_Match_Score", ascending=False).head(top_n + 1)

    # Add Rank column (1-based) after final sorting
    out = out.reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))

    cols = [
        "Rank",
        "Candidate_ID",
        "Name",
        "Email",
        "Experience_Years",
        "Skills",
        "Category",
        "Final_Match_Score",
        "Text_Similarity",
        "Experience_Score",
    ]
    present_cols = [c for c in cols if c in out.columns]
    return out[present_cols]

# ------------------------- UI Functions -------------------------
def show_dataset_analysis():
    """Display dataset analysis with individual graphs in sequence"""
    df, _, _, _ = get_cached_data()
    if df is None:
        st.warning("Dataset not loaded. Check CSV_PATH and file existence.")
    else:
        st.write("### Shape of Dataset")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.write("### First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)

        # 1. Candidate Categories Distribution
        st.write("### Candidate Categories Distribution")
        try:
            fig, ax = plt.subplots()
            sns.countplot(x="Category", data=df, order=df["Category"].value_counts().index, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Cannot plot category distribution: {e}")

        # 2. Experience Distribution
        st.write("### Experience Distribution")
        try:
            fig, ax = plt.subplots()
            sns.histplot(df["Experience_Years"], bins=20, kde=True, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Cannot plot experience distribution: {e}")

        # 3. Experience by Category (Boxplot)
        st.write("### Experience by Category (Boxplot)")
        try:
            fig, ax = plt.subplots()
            sns.boxplot(x="Category", y="Experience_Years", data=df, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Cannot plot boxplot: {e}")

        # 4. Top Skills (Frequency)
        st.write("### Top Skills (Frequency)")
        try:
            skills_series = df.get("Skills", pd.Series(dtype=str)).fillna("").astype(str)
            tokens = []
            for row in skills_series:
                tokens.extend([t.strip().lower() for t in row.split(",") if t.strip()])
            if tokens:
                vc = pd.Series(tokens).value_counts().head(20)
                fig, ax = plt.subplots(figsize=(6, 5))
                vc.sort_values().plot(kind="barh", ax=ax, color="#60a5fa")
                ax.set_xlabel("Count")
                ax.set_ylabel("Skill")
                st.pyplot(fig)
            else:
                st.info("No skills found to plot.")
        except Exception as e:
            st.warning(f"Cannot plot skills frequency: {e}")

        # 5. Email Domains (Top 10)
        st.write("### Email Domains (Top 10)")
        try:
            domains = (
                df.get("Email", pd.Series(dtype=str))
                  .dropna().astype(str).str.extract(r"@(.+)")[0].str.lower()
            )
            vc = domains.value_counts().head(10)
            if not vc.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                vc.sort_values().plot(kind="barh", ax=ax, color="#34d399")
                ax.set_xlabel("Count")
                ax.set_ylabel("Domain")
                st.pyplot(fig)
            else:
                st.info("No email domains to plot.")
        except Exception as e:
            st.warning(f"Cannot plot email domains: {e}")

        # 6. Correlation Heatmap
        st.write("### Correlation Heatmap")
        try:
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Cannot plot correlation heatmap: {e}")

        # 7. Category Share (Pie)
        st.write("### Category Share (Pie)")
        try:
            vc = df["Category"].fillna("Unknown").value_counts()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Cannot plot category share: {e}")

def show_talent_ai():
    """Display the talent AI functionality"""
    df, _, _, _ = get_cached_data()
    if df is None:
        st.error("Dataset not loaded. Please check your CSV file.")
        return
   
    sample_count = len(df)
    categories = get_categories(df)

    with st.sidebar:
        st.header("Filters")
        top_n = st.number_input("Top N candidates", min_value=10, max_value=1000, value=TOP_N_DEFAULT, step=10)
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            wt_text = st.number_input("Weight: Text", min_value=0.0, max_value=1.0, value=0.85, step=0.05)
        with col_w2:
            wt_exp = st.number_input("Weight: Experience", min_value=0.0, max_value=1.0, value=0.15, step=0.05)

        min_exp = st.text_input("Min experience (years)")
        max_exp = st.text_input("Max experience (years)")
        selected_categories = st.multiselect("Categories", options=categories)
        include_skills = st.text_input("Include skills (comma separated)")
        exclude_skills = st.text_input("Exclude skills (comma separated)")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Preset JD chips
        st.markdown("*Preset JDs* — click to autofill:")
        chip_cols = st.columns(5)
        jd_presets = [
            ("Data\nScientist", "Looking for a Data Scientist with strong Python, SQL, Machine Learning, NLP experience to build models and dashboards."),
            ("Backend\nDeveloper", "Hiring a Backend Developer skilled in Java, Spring Boot, Microservices, REST APIs, SQL and cloud deployment."),
            ("Frontend\nDeveloper", "Seeking a Frontend Developer with React, JavaScript, HTML, CSS, responsive design and testing skills."),
            ("Data\nAnalyst", "Need a Data Analyst proficient in Excel, SQL, Tableau/Power BI, data cleaning and visualization."),
            ("DevOps\nEngineer", "Looking for a DevOps Engineer with Docker, Kubernetes, CI/CD, AWS/Azure and Terraform."),
        ]
        for (col, (label, text_val)) in zip(chip_cols, jd_presets):
            with col:
                if st.button(label, key=f"chip_{label}"):
                    st.session_state["jd_text"] = text_val

        jd = st.text_area(
            "Job Description",
            placeholder="Looking for a Data Scientist with Python, SQL, Machine Learning, NLP experience...",
            height=220,
            key="jd_text",
        )

        uploaded_resume_data = st.session_state.get("uploaded_resume_data")

        run = st.button("Rank Candidates", type="primary")

        if run:
            if not jd.strip():
                st.error("Please enter a Job Description")
            else:
                results = rank_candidates(
                    job_desc=jd.strip(),
                    top_n=int(top_n),
                    wt_text=float(wt_text),
                    wt_exp=float(wt_exp),
                    min_exp=min_exp,
                    max_exp=max_exp,
                    include_skills=include_skills,
                    exclude_skills=exclude_skills,
                    selected_categories=selected_categories,
                    uploaded_resume=uploaded_resume_data,
                )

                st.subheader("Top Ranked Candidates")
                st.dataframe(results, use_container_width=True)

                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name="ranked_candidates.csv",
                    mime="text/csv",
                )

                st.markdown(f"Rows in dataset: *{sample_count}*")

    with col_right:
        st.subheader("Upload Resume PDF")
        st.caption(f"Maximum file size: 200MB")
       
        file = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False, help="Maximum file size: 200MB")
        if file is not None:
            try:
                # Check file size before processing
                if file.size > MAX_PDF_SIZE:
                    st.error(f"File size ({file.size / (1024*1024):.1f}MB) exceeds maximum allowed size (200MB)")
                else:
                    # Read into a stable buffer to ensure proper parsing
                    raw_bytes = file.read()
                    buf = io.BytesIO(raw_bytes)
                    text = extract_text_from_pdf(buf)
                    insights = extract_insights_from_resume(text)
                   
                    # Fallback summary if NLP summary is empty
                    if not insights.get("summary") and text:
                        compact = " ".join([ln.strip() for ln in text.splitlines() if ln.strip()])
                        insights["summary"] = compact[:300]
                   
                    if jd:
                        insights["resume_score"] = calculate_resume_score(insights, jd)
                    else:
                        insights["resume_score"] = 0

                    st.session_state["uploaded_resume_data"] = {
                        "insights": insights,
                        "text": text,
                        "filename": file.name,
                    }

                    st.success("Resume analyzed successfully!")
                    st.metric("Resume Score", f"{insights.get('resume_score', 0)}")
                    st.write("Name:", insights.get("name") or "Not found")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("Experience:", f"{insights.get('experience_years', 0)} years")
                    with col_b:
                        st.write("Skills found:", len(insights.get("skills", [])))
                    if insights.get("skills"):
                        badges_html = "".join([f"<span class='badge'>{re.escape(sk).replace(chr(92), '')}</span>" for sk in insights["skills"][:12]])
                        st.markdown(badges_html, unsafe_allow_html=True)
                    if insights.get("summary"):
                        st.markdown("<div class='soft-card' style='margin-top:8px'><b>Summary</b><br/>" + insights["summary"].replace("<", "&lt;").replace(">", "&gt;") + "</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Upload failed: {e}")

        if st.session_state.get("uploaded_resume_data"):
            st.info("This uploaded resume will be included when you rank candidates.")

    with st.expander("Dataset Status"):
        st.write("File:", os.path.basename(CSV_PATH))
        st.write("Rows:", sample_count)
        st.write("Vectorizer: TF-IDF (1-2 grams, min_df=2)")
        df, _, _, report = get_cached_data()
        if report:
            st.write("Validation Summary")
            st.json(report)

# ------------------------- Main App -------------------------
def main():
    st.set_page_config(page_title="Talent AI - Smart Resume Screening", layout="wide")
   
    # Note: Removed deprecated Streamlit option 'deprecation.showfileUploaderEncoding'

    # Inject lightweight CSS
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"], .stMarkdown, .stText, .stButton>button { font-family: 'Inter', sans-serif; }
        .soft-card {background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 16px; padding: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.06);}
        .chip {display:inline-block; padding:6px 10px; border-radius:8px; background:#f1f5f9; color:#0f172a; margin:4px; cursor:pointer; font-size:13px; border:1px solid #e2e8f0;}
        .chip:hover {background:#e2e8f0;}
        .badge {display:inline-block; padding:4px 8px; border-radius:8px; background:#eef2ff; color:#3730a3; margin:2px; font-size:12px;}
        .metric-box {background:#ecfeff; border:1px solid #cffafe; border-radius:10px; padding:10px;}
        .app-header {position: sticky; top: 0; z-index: 10; background: rgba(255,255,255,0.8); backdrop-filter: blur(8px); border-bottom: 1px solid #e5e7eb;}
        .header-inner {display:flex; align-items:center; justify-content:space-between; padding: 12px 8px;}
        .logo {height:40px; width:40px; border-radius:16px; display:flex; align-items:center; justify-content:center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); position:relative; overflow:hidden;}
        .logo:before {content:''; position:absolute; top:-50%; left:-50%; width:200%; height:200%; background: linear-gradient(45deg, transparent, rgba(255,255,255,0.12), transparent); animation: shimmer 3s infinite;}
        @keyframes shimmer { 0% { transform: translateX(-100%) translateY(-100%) rotate(45deg);} 100% { transform: translateX(100%) translateY(100%) rotate(45deg);} }
        .title {font-weight: 800; font-size: 22px; background: linear-gradient(90deg, #2563eb, #7c3aed); -webkit-background-clip: text; background-clip: text; color: transparent; margin:0;}
        .subtitle {font-size: 12px; color: #64748b; margin:0;}
        .stButton>button { border-radius: 12px; padding: 15px 20px; background: linear-gradient(90deg, #2563eb, #7c3aed); color: #fff; border: none; box-shadow: 0 8px 18px rgba(37,99,235,0.25); white-space: pre-line; font-size: 14px; min-height: 60px; }
        .stButton>button:hover { filter: brightness(0.95); }
        /*Equal Width for preset JD button */
        .stButton {width:100%;}
        .stButton button{width:100%; min-width:120px;}
        /* Hide the default 200MB limit text */
        .stFileUploader > div > div > div > div > small { display: none !important; }
        .stFileUploader > div > div > div > div > div > small { display: none !important; }
        .stFileUploader small { display: none !important; }
        .stFileUploader div[data-testid="stFileUploader"] small { display: none !important; }
        /* Additional targeting for file uploader limit text */
        .stFileUploader > div > div > div > div > div > div > small { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Font Awesome for icons
    st.markdown("<link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css\">", unsafe_allow_html=True)

    # Header
    st.markdown(
        """
        <div class="app-header">
          <div class="header-inner">
            <div style="display:flex; align-items:center; gap:12px;">
              <div class="logo"><i class="fas fa-robot" style="color:white; position:relative; z-index:1;"></i></div>
              <div>
                <div class="title">Talent AI</div>
                <div class="subtitle">Smart Resume Screening & Ranking</div>
              </div>
            </div>
            <div class="subtitle" style="display:flex; align-items:center; gap:6px;">
              <i class="fas fa-brain" style="color:#3b82f6;"></i>
              AI-Powered • TF-IDF + Cosine
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Create tabs for navigation
    tab1, tab2 = st.tabs(["🤖 Talent AI", "📊 Data Analysis"])
   
    with tab1:
        show_talent_ai()
   
    with tab2:
        show_dataset_analysis()
 
if __name__ == "__main__":
    main()