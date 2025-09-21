# 🚀 Deployment Guide for Talent AI

## Quick Deployment Steps

### 1. Push to GitHub
```bash
# Add the new files
git add nltk.txt DEPLOYMENT.md
git commit -m "Add NLTK data file and deployment guide"
git push origin main
```

### 2. Deploy on Streamlit Cloud

1. **Go to [Streamlit Cloud](https://share.streamlit.io)**
2. **Sign in with GitHub**
3. **Click "New app"**
4. **Select your repository**: `yourusername/ai-resume-screening`
5. **Configure**:
   - **Main file path**: `app.py`
   - **Branch**: `main`
   - **Python version**: 3.8+
6. **Click "Deploy"**

### 3. Your App Will Be Live At:
`https://your-app-name.streamlit.app`

## Files Included for Deployment

- ✅ `app.py` - Main Streamlit application
- ✅ `requirements.txt` - Python dependencies
- ✅ `candidates.csv` - Sample dataset (50,000+ candidates)
- ✅ `nltk.txt` - NLTK data requirements for cloud deployment
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `.gitignore` - Git ignore rules
- ✅ `README.md` - Project documentation

## Troubleshooting

### If deployment fails:
1. Check that all dependencies are in `requirements.txt`
2. Ensure `nltk.txt` includes required NLTK data
3. Verify `app.py` runs locally with `streamlit run app.py`

### If NLTK data is missing:
The `nltk.txt` file ensures NLTK downloads the required data packages automatically during deployment.

## Features Ready for Production

- 🤖 AI-powered candidate ranking
- 📄 PDF resume analysis
- 📊 Interactive data visualization
- 🔍 Advanced filtering options
- 📥 CSV export functionality
- 📱 Responsive design

Your app is now ready for production deployment!
