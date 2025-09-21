# 🔧 Troubleshooting Guide

## Common Streamlit Cloud Deployment Issues

### ❌ "installer returned a non-zero exit code" Error

**This error usually means:**
- Dependency version conflicts
- Missing system packages
- Python version incompatibility

**✅ I've Fixed This By:**
1. **Updated requirements.txt** with compatible version ranges
2. **Added runtime.txt** to specify Python 3.9.18
3. **Added packages.txt** for system dependencies
4. **Updated nltk.txt** with required NLTK data

### 🚀 How to Deploy the Fixed Version:

1. **Push the updated files:**
   ```bash
   git add .
   git commit -m "Fix dependency installation issues"
   git push origin main
   ```

2. **Redeploy on Streamlit Cloud:**
   - Go to your app on [share.streamlit.io](https://share.streamlit.io)
   - Click "Reboot app" or "Deploy" again
   - The new requirements will be installed

### 🔍 If Still Having Issues:

**Check the logs:**
1. Go to your Streamlit Cloud app
2. Click on "Manage app"
3. Check the "Logs" tab for specific error messages

**Common fixes:**
- If NLTK errors: The nltk.txt file should handle this
- If matplotlib errors: Pillow is now included
- If scikit-learn errors: Version ranges should resolve conflicts

### 📋 Files Added for Fixes:
- ✅ `runtime.txt` - Python version specification
- ✅ `packages.txt` - System dependencies
- ✅ Updated `requirements.txt` - Compatible versions
- ✅ Updated `nltk.txt` - Complete NLTK data

### 🆘 Still Need Help?
If you're still getting errors, share the specific error message from the Streamlit Cloud logs and I'll help you fix it!
