@echo off
echo ========================================
echo    Push Dependency Fixes to GitHub
echo ========================================
echo.

echo I've fixed the dependency installation issues!
echo Now you need to push these fixes to GitHub.
echo.

set /p REPO_URL="Enter your GitHub repository URL (e.g., https://github.com/username/repo-name.git): "

echo.
echo Adding remote origin...
git remote add origin %REPO_URL%

echo.
echo Renaming branch to main...
git branch -M main

echo.
echo Pushing fixes to GitHub...
git push -u origin main

echo.
echo ========================================
echo    SUCCESS! Fixes pushed to GitHub!
echo ========================================
echo.
echo Now go to Streamlit Cloud and redeploy your app!
echo The dependency errors should be fixed now.
echo.
pause
