# 🤖 Talent AI: Smart Resume Screening and Ranking System

An AI-powered resume screening application that uses advanced semantic matching to intelligently rank candidates based on job descriptions. Features hybrid BERT + TF-IDF matching with persistent disk caching for instant searches.

## ✨ Features

- **🧠 Hybrid Semantic Matching**: Combines BERT (deep learning) + TF-IDF (keyword matching) for best results
- **⚡ Three Matching Modes**: 
  - **Hybrid** (default): Best of both worlds - semantic understanding + keyword precision
  - **BERT Only**: Pure semantic matching - finds candidates regardless of wording
  - **TF-IDF Only**: Fast keyword matching - instant results
- **💾 Persistent Disk Caching**: Embeddings saved to disk for fast startup (~10 seconds)
- **📄 PDF Resume Analysis**: Upload and analyze PDF resumes with automatic text extraction
- **🎯 Advanced Filtering**: Filter by experience, skills, categories, and more
- **📊 Interactive Dashboard**: Beautiful, responsive Flask UI with real-time results
- **📈 Data Analysis**: Comprehensive dataset analysis with charts and statistics
- **💼 Export Functionality**: Download ranked candidate lists as CSV files

## 🚀 Live Demo

Access the application at `http://localhost:5000` after running locally.

## 🛠️ Technology Stack

- **Frontend**: HTML, CSS, JavaScript (Bootstrap)
- **Backend**: Flask (Python)
- **ML Libraries**: 
  - **sentence-transformers** (BERT) - Semantic understanding
  - **scikit-learn** - TF-IDF vectorization and similarity
  - **torch** - Deep learning backend
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **PDF Processing**: PyPDF2
- **Caching**: NumPy compressed arrays (.npz) for persistent storage

## 📋 Prerequisites

- Python 3.8 or higher
- Git

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-resume-screening.git
   cd ai-resume-screening
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **First-time setup: Pre-compute BERT embeddings** (One-time, 3-5 minutes)
   ```bash
   precompute_embeddings.bat  # On Windows
   ```
   This creates a disk cache of embeddings for all 50,000 candidates.

5. **Run the application**
   ```bash
   run_app.bat  # On Windows
   # Or manually:
   # set SKIP_BERT_PRECOMPUTE=true
   # python app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5000`

### Startup Times

- **First time** (with pre-computation): 3-5 minutes (one-time only)
- **Subsequent startups**: ~10 seconds (loads from disk cache)
- **Searches**: Instant (embeddings pre-loaded)

## 📁 Project Structure

```
ai-resume-screening/
├── app.py                          # Main Flask application
├── bert_matcher.py                 # BERT semantic matching module
├── candidates.csv                  # Sample candidate dataset (50K candidates)
├── requirements.txt                # Python dependencies
├── run_app.bat                     # Windows startup script
├── precompute_embeddings.bat       # One-time BERT cache setup
├── .bert_cache/                    # Persistent embedding cache (auto-created)
│   ├── embeddings.npz              # Compressed BERT embeddings
│   └── metadata.json               # Embedding metadata
├── static/                         # CSS and JavaScript files
│   └── style.css
├── templates/                      # HTML templates
│   ├── index.html                  # Main search page
│   ├── analysis.html               # Data analysis page
│   └── base.html                   # Base template
├── .streamlit/
│   └── config.toml                 # Configuration
├── .gitignore
└── README.md                       # This file
```

## 🔧 Configuration

The application uses environment variables for configuration:

- `CANDIDATE_CSV`: Path to the candidate CSV file (default: "candidates.csv")
- `TOP_N_DEFAULT`: Default number of top candidates to show (default: 100)
- `SKIP_BERT_PRECOMPUTE`: Skip pre-computation at startup (default: "true" - loads from disk)
- `BERT_MODEL_NAME`: BERT model to use (default: "sentence-transformers/all-MiniLM-L6-v2")
- `BERT_BATCH_SIZE`: Batch size for encoding (default: 8)
- `BERT_DEVICE`: Device to use - "cpu", "cuda", or "auto" (default: "auto")
- `MATCHING_MODE`: Default matching mode - "hybrid", "bert", or "tfidf" (default: "hybrid")

## 📊 Dataset Format

The application expects a CSV file with the following columns:

- `Candidate_ID`: Unique identifier for each candidate
- `Name`: Candidate's full name
- `Email`: Candidate's email address
- `Experience_Years`: Years of professional experience (numeric)
- `Skills`: Comma-separated list of skills
- `Category`: Job category (e.g., "Software Engineering", "Data Science")
- `Resume_Summary`: Brief summary of the candidate's background

## 🎯 How It Works

### Matching Modes

1. **Hybrid Mode (Recommended)**
   - Combines BERT semantic understanding + TF-IDF keyword matching
   - Formula: `(BERT_score + TF-IDF_score) / 2`
   - Best for: Balanced results with both semantic and keyword precision

2. **BERT Only Mode**
   - Pure semantic matching using deep learning
   - Understands meaning, not just keywords
   - Finds candidates who describe skills differently
   - Example: "programmer" matches "software engineer"

3. **TF-IDF Only Mode**
   - Fast keyword-based matching
   - Instant results, no BERT encoding needed
   - Best for: Exact keyword matching

### Processing Pipeline

1. **Text Processing**: Cleans and normalizes text data
2. **Vectorization**: 
   - **BERT**: Converts text to 384-dimensional semantic embeddings
   - **TF-IDF**: Creates sparse keyword vectors
3. **Caching**: Saves BERT embeddings to disk (`.bert_cache/`)
4. **Similarity Calculation**: Computes cosine similarity
5. **Scoring**: Combines text similarity + experience scores (weighted)
6. **Ranking**: Sorts candidates by final match score

### Disk Caching System

- **First run**: Pre-computes embeddings for all candidates (3-5 minutes)
- **Saves to disk**: `.bert_cache/embeddings.npz` (~50-100 MB)
- **Subsequent runs**: Loads from disk in ~10 seconds
- **Persistent**: Survives restarts, no re-computation needed

## 🔍 Features in Detail

### Main Search Page
- **Job Description Input**: Enter job requirements or use templates
- **Matching Mode Selector**: Choose Hybrid, BERT Only, or TF-IDF Only
- **PDF Resume Upload**: Analyze and compare uploaded resumes
- **Advanced Filters**:
  - Experience range (min/max years)
  - Required skills (must have all)
  - Excluded skills (must not have any)
  - Category selection
- **Weighted Scoring**: Adjust text vs experience importance
- **Results Display**: 
  - Shows BERT similarity, TF-IDF similarity (in Hybrid mode)
  - Experience score and final match score
  - Candidate details with contact information
- **Export**: Download results as CSV

### Data Analysis Page
- Dataset overview and statistics
- Category distribution charts
- Experience distribution analysis
- Skills frequency analysis
- Email domain analysis
- Correlation heatmaps

## 🔄 Updating the Cache

If you update `candidates.csv` with new candidates:

```bash
# Re-run pre-computation
precompute_embeddings.bat

# Or delete cache and restart
rmdir /s /q .bert_cache
python app.py  # Will pre-compute on startup
```

## � Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Additional Documentation

Documentation for advanced features is available in the code comments and docstrings.

## 🐛 Troubleshooting

### "No cached embeddings found on disk"
This is normal on first run. Run `precompute_embeddings.bat` to create the cache.

### BERT/Hybrid mode shows no results
Make sure you've run pre-computation first. Check if `.bert_cache/` directory exists with `embeddings.npz` and `metadata.json`.

### Slow startup
If `SKIP_BERT_PRECOMPUTE=false`, it will pre-compute at startup (3-5 minutes). Set it to `true` to load from disk instead.

### Out of memory
Reduce `BERT_BATCH_SIZE` environment variable (default: 8) to 4 or 2.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **sentence-transformers** - For the excellent BERT implementation
- **HuggingFace** - For the pre-trained models
- **Flask** - For the web framework
- **scikit-learn** - For TF-IDF and similarity calculations