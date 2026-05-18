---
title: Talent AI Smart Resume Screening
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Talent AI — Smart Resume Screening & Ranking System

An AI-powered resume screening application that ranks candidates based on job descriptions using BERT and TF-IDF matching.

## Features

- **BERT + TF-IDF Hybrid Matching** — semantic similarity using sentence-transformers
- **PDF Resume Upload** — upload and compare your resume against the candidate pool
- **Advanced Filters** — filter by experience, skills, category
- **Analytics Dashboard** — visualize candidate distribution and skill trends
- **CSV Export** — download ranked results

## Usage

1. Enter a job description
2. Adjust matching weights and filters
3. Click **Rank Candidates**
4. Optionally upload your own resume to see how you compare

## Tech Stack

- Python / Flask
- sentence-transformers (all-MiniLM-L6-v2)
- scikit-learn TF-IDF
- pandas, numpy, matplotlib
- gunicorn (production WSGI server)
