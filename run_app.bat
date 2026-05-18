@echo off
echo ======================================================================
echo Starting Flask App with BERT Integration (Disk Cache Enabled)
echo ======================================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Set environment variable to skip pre-computation (will load from disk instead)
set SKIP_BERT_PRECOMPUTE=true
echo Environment: SKIP_BERT_PRECOMPUTE=%SKIP_BERT_PRECOMPUTE%
echo.
echo If .bert_cache/ exists, embeddings will load from disk in ~10 seconds
echo Otherwise, embeddings will be computed on first BERT/Hybrid search
echo ======================================================================
echo.

REM Run the Flask app
python app.py

pause
