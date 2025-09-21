Write-Host "========================================" -ForegroundColor Cyan
Write-Host "    Talent AI - GitHub Push Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Please follow these steps:" -ForegroundColor Yellow
Write-Host "1. Create a GitHub repository first" -ForegroundColor White
Write-Host "2. Copy the repository URL" -ForegroundColor White
Write-Host "3. Paste it when prompted below" -ForegroundColor White
Write-Host ""

$repoUrl = Read-Host "Enter your GitHub repository URL (e.g., https://github.com/username/repo-name.git)"

Write-Host ""
Write-Host "Adding remote origin..." -ForegroundColor Green
git remote add origin $repoUrl

Write-Host ""
Write-Host "Renaming branch to main..." -ForegroundColor Green
git branch -M main

Write-Host ""
Write-Host "Pushing to GitHub..." -ForegroundColor Green
git push -u origin main

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "    SUCCESS! Your code is now on GitHub!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next step: Go to https://share.streamlit.io to deploy your app!" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to continue"
