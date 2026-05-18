@echo off
echo ======================================================================
echo PRE-COMPUTING BERT EMBEDDINGS (One-Time Setup with Disk Cache)
echo ======================================================================
echo This will:
echo   1. Encode all 50,000 candidates with BERT (3-5 minutes)
echo   2. Save embeddings to disk in .bert_cache/ directory
echo   3. Enable instant BERT/Hybrid searches on future startups
echo.
echo You only need to run this ONCE. After this, embeddings will load
echo from disk in ~10 seconds on every startup!
echo ======================================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run pre-computation script
python test_disk_cache.py

echo.
echo ======================================================================
echo Press any key to exit...
pause
