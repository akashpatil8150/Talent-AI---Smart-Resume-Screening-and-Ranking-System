import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import re
import os

# Simple version for testing deployment
st.set_page_config(page_title="Talent AI - Simple Version", layout="wide")

st.title("🤖 Talent AI - Resume Screening")
st.write("This is a simplified version for testing deployment.")

# Check if CSV exists
if os.path.exists("candidates.csv"):
    st.success("✅ CSV file found!")
    df = pd.read_csv("candidates.csv")
    st.write(f"Dataset loaded: {len(df)} candidates")
    st.dataframe(df.head())
else:
    st.error("❌ CSV file not found")
    st.write("Please ensure candidates.csv is in the same directory as app.py")

st.write("If you can see this message, the basic deployment is working!")
