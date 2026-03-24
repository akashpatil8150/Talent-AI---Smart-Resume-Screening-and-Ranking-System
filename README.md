# 🤖 Talent AI - Smart Resume Screening & Ranking System

An AI-powered resume screening application that uses machine learning to intelligently rank candidates based on job descriptions and uploaded resumes.

## ✨ Features

- **Smart Candidate Ranking**: Uses TF-IDF vectorization and cosine similarity to match candidates with job descriptions
- **PDF Resume Analysis**: Upload and analyze PDF resumes with automatic text extraction and insights
- **Advanced Filtering**: Filter candidates by experience, skills, categories, and more
- **Interactive Dashboard**: Beautiful, responsive UI with real-time data visualization
- **Data Analysis**: Comprehensive dataset analysis with charts and statistics
- **Export Functionality**: Download ranked candidate lists as CSV files

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://talent-ai---smart-resume-screening-and-ranking-system-culv5cse.streamlit.app)

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Libraries**: scikit-learn, NLTK
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **PDF Processing**: PyPDF2

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
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

### Deploy on Streamlit Cloud

1. **Fork this repository** on GitHub

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**

3. **Click "New app"**

4. **Connect your GitHub account** and select your forked repository

5. **Configure the app**:
   - **Main file path**: `app.py`
   - **Branch**: `main` (or your default branch)
   - **Python version**: 3.8+

6. **Click "Deploy"**

## 📁 Project Structure

```
ai-resume-screening/
├── app.py                 # Main Streamlit application
├── candidates.csv         # Sample candidate dataset
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## 🔧 Configuration

The application uses environment variables for configuration:

- `CANDIDATE_CSV`: Path to the candidate CSV file (default: "candidates.csv")
- `TOP_N_DEFAULT`: Default number of top candidates to show (default: 100)
- `MAX_FEATURES`: Maximum features for TF-IDF vectorizer (default: 150000)
- `MIN_DF`: Minimum document frequency for TF-IDF (default: 2)

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

1. **Text Processing**: Cleans and normalizes text data from job descriptions and resumes
2. **Vectorization**: Uses TF-IDF to convert text into numerical vectors
3. **Similarity Calculation**: Computes cosine similarity between job descriptions and candidate profiles
4. **Scoring**: Combines text similarity with experience scores using weighted averages
5. **Ranking**: Ranks candidates based on their final match scores

## 🔍 Features in Detail

### Talent AI Tab
- Enter job descriptions or use preset templates
- Upload PDF resumes for analysis
- Apply various filters (experience, skills, categories)
- View ranked candidate results
- Download results as CSV

### Data Analysis Tab
- Dataset overview and statistics
- Category distribution charts
- Experience distribution analysis
- Skills frequency analysis
- Email domain analysis
- Correlation heatmaps

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
