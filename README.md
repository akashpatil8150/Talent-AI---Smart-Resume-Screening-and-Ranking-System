
# Talent AI — Smart Resume Screening & Ranking System

An AI-powered resume screening application that ranks candidates based on job descriptions using a hybrid BERT + TF-IDF matching engine.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-blue?logo=huggingface)](https://akash8150-talent-ai-smart-resume-screening-and-r-175d457.hf.space/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/akashpatil8150/Talent-AI---Smart-Resume-Screening-and-Ranking-System)

---

## 🚀 Live Demo

👉 **[Try it here](https://akash8150-talent-ai-smart-resume-screening-and-r-175d457.hf.space/)**

---

## 📌 About the Project

Hiring teams spend hours manually reviewing resumes. Talent AI automates that process by intelligently ranking a pool of candidates against any job description you provide — in seconds.

The system combines two complementary NLP techniques: **TF-IDF** for keyword-level matching and **BERT sentence embeddings** for deep semantic understanding. This means it doesn't just look for exact keyword matches — it understands the *meaning* behind the job description and finds candidates whose profiles are semantically aligned, even if they use different terminology.

Beyond ranking the existing candidate pool, you can also **upload your own resume as a PDF** and instantly see how you stack up against other candidates for a given role.

---

## ✨ Features

- **Hybrid BERT + TF-IDF Matching** — combines semantic similarity (sentence-transformers) with keyword relevance (TF-IDF) for more accurate rankings
- **Three Matching Modes** — choose between `bert`, `tfidf`, or `hybrid` depending on your needs
- **PDF Resume Upload** — upload your own resume and compare yourself against the candidate pool
- **Resume Insight Extraction** — automatically extracts name, email, phone, skills, experience, and education from uploaded PDFs
- **Advanced Filters** — filter candidates by experience range, required skills, and job category
- **Analytics Dashboard** — visualize candidate score distributions, skill trends, and category breakdowns using matplotlib/seaborn charts
- **CSV Export** — download the full ranked results as a CSV file
- **Embedding Cache** — pre-computed BERT embeddings are saved to disk so repeated queries are fast

---

## 🧠 How It Works

### 1. Data Loading & Preprocessing

When the app starts, it loads `candidates.csv` which contains candidate profiles with fields like `Name`, `Skills`, `Resume_Summary`, `Experience_Years`, and `Category`. Each candidate's skills and resume summary are combined into a single text field and cleaned (lowercased, URLs removed, special characters stripped).

Experience years are normalized to a `[0, 1]` scale using `MinMaxScaler` so they can be factored into the final score alongside text similarity.

### 2. TF-IDF Vectorization

A `TfidfVectorizer` is built over all candidate text using unigrams and bigrams (`ngram_range=(1,2)`) with up to 150,000 features. This creates a sparse matrix representing each candidate's profile as a weighted term vector.

When you enter a job description, it's transformed using the same vectorizer and cosine similarity is computed against every candidate — giving a keyword-relevance score for each.

### 3. BERT Semantic Matching

The app uses the `sentence-transformers/all-MiniLM-L6-v2` model to generate dense 384-dimensional embeddings for both the job description and each candidate's combined text.

Cosine similarity between the job description embedding and each candidate embedding gives a **semantic similarity score** — capturing meaning beyond exact keyword overlap. For example, a candidate who writes "built ML pipelines" will still match a job description that says "machine learning engineer" even without identical wording.

Candidate embeddings are pre-computed and cached to disk (`.bert_cache/embeddings.npz`) so they don't need to be recomputed on every query.

### 4. Hybrid Scoring

The final candidate score is a weighted combination:

```
final_score = (text_similarity_weight) × text_sim + (experience_weight) × exp_score
```

In `hybrid` mode, `text_sim` is the average of the BERT and TF-IDF similarity scores. In `bert` or `tfidf` mode, only the respective score is used. The experience score is the normalized years of experience.

Default weights: **85% text similarity + 15% experience score**.

### 5. PDF Resume Analysis

When you upload a PDF resume, the app:
1. Extracts raw text using `PyPDF2`
2. Runs regex-based NLP to pull out name, email, phone, skills, experience years, and education
3. Encodes the resume text using BERT and computes similarity against the job description
4. Calculates a composite score (40% skill match, 30% experience, 20% completeness, 10% keyword relevance)
5. Shows where your resume ranks in the candidate pool

### 6. Filtering & Ranking

After scoring, candidates can be filtered by:
- Minimum / maximum experience years
- Required skills (comma-separated)
- Job category

Results are sorted by final score (descending) and the top N candidates are returned.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Flask Web App                     │
│                      (app.py)                        │
├──────────────────────┬──────────────────────────────┤
│   TF-IDF Engine      │      BERT Engine              │
│  (scikit-learn)      │   (bert_matcher.py)           │
│                      │                               │
│  TfidfVectorizer     │  BERTEncoder                  │
│  Cosine Similarity   │  SimilarityCalculator         │
│                      │  EmbeddingCache               │
├──────────────────────┴──────────────────────────────┤
│              HybridScorer (bert_matcher.py)          │
│         Weighted combination of both scores          │
├─────────────────────────────────────────────────────┤
│           candidates.csv  │  PDF Upload              │
│           (candidate pool)│  (PyPDF2 + NLP)          │
└─────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Python / Flask |
| Semantic Matching | sentence-transformers (`all-MiniLM-L6-v2`) |
| Keyword Matching | scikit-learn TF-IDF |
| PDF Parsing | PyPDF2 |
| NLP Utilities | NLTK |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Production Server | gunicorn |
| Containerization | Docker |

---

## ⚙️ Configuration

Key settings can be controlled via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MATCHING_MODE` | `hybrid` | `bert`, `tfidf`, or `hybrid` |
| `BERT_MODEL_NAME` | `all-MiniLM-L6-v2` | HuggingFace model to use |
| `BERT_FORCE_CPU` | `true` | Force CPU inference |
| `SKIP_BERT_PRECOMPUTE` | `true` | Skip embedding pre-computation at startup |
| `TOP_N_DEFAULT` | `100` | Default number of results to return |
| `MAX_FEATURES` | `150000` | TF-IDF vocabulary size |
| `CANDIDATE_CSV` | `candidates.csv` | Path to candidate dataset |

Copy `.env.example` to `.env` and adjust as needed.

---

## 🚀 Getting Started Locally

```bash
git clone https://github.com/akashpatil8150/Talent-AI---Smart-Resume-Screening-and-Ranking-System.git
cd Talent-AI---Smart-Resume-Screening-and-Ranking-System

pip install -r requirements.txt

# Optional: pre-compute BERT embeddings for faster queries
precompute_embeddings.bat   # Windows
# or: python -c "from app import initialize_bert; initialize_bert()"

python app.py
```

Then open `http://localhost:5000` in your browser.

---

## 🐳 Docker

```bash
docker build -t talent-ai .
docker run -p 7860:7860 talent-ai
```

---

## 📁 Project Structure

```
├── app.py                  # Main Flask application
├── bert_matcher.py         # BERT encoding, similarity, and caching
├── candidates.csv          # Candidate dataset
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
├── .bert_cache/            # Pre-computed BERT embeddings (auto-generated)
├── static/style.css        # Frontend styles
└── templates/              # Jinja2 HTML templates
    ├── base.html
    ├── index.html
    └── analysis.html
```

---

## 📄 License
