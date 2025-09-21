@echo off
echo ========================================
echo    Talent AI - GitHub Push Script
echo ========================================
echo.

echo Please follow these steps:
echo 1. Create a GitHub repository first
echo 2. Copy the repository URL
echo 3. Paste it when prompted below
echo.

set /p REPO_URL="Enter your GitHub repository URL (e.g., https://github.com/username/repo-name.git): "

echo.
echo Adding remote origin...
git remote add origin %REPO_URL%

echo.
echo Renaming branch to main...
git branch -M main

echo.
echo Pushing to GitHub...
git push -u origin main

echo.
echo ========================================
echo    SUCCESS! Your code is now on GitHub!
echo ========================================
echo.
echo Next step: Go to https://share.streamlit.io to deploy your app!
echo.
pause
