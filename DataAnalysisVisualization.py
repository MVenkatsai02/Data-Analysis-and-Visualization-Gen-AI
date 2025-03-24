import streamlit as st 
import pandas as pd
import numpy as np
import json
import google.generativeai as genai
import chardet  # To detect file encoding
import faiss
import pickle
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
if "ai_history" not in st.session_state:
    st.session_state.ai_history = []
if "preprocessing_selections" not in st.session_state:
    st.session_state.preprocessing_selections = {}
if "faiss_index" not in st.session_state:
    d = 512  # Dimension size for FAISS index
    st.session_state.faiss_index = faiss.IndexFlatL2(d)
    st.session_state.embeddings = []  # Store embeddings separately

def load_data(file):
    """Load data from uploaded file with proper encoding handling."""
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        try:
            return pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            file.seek(0)
            raw_data = file.read()
            detected_encoding = chardet.detect(raw_data)['encoding']
            file.seek(0)
            try:
                return pd.read_csv(file, encoding=detected_encoding)
            except UnicodeDecodeError:
                for encoding in ["ISO-8859-1", "latin1", "cp1252"]:
                    file.seek(0)
                    try:
                        return pd.read_csv(file, encoding=encoding)
                    except UnicodeDecodeError:
                        continue
                st.error("Unable to read the CSV file due to encoding issues.")
                return None
    
    elif file_extension == 'xlsx':
        try:
            return pd.read_excel(file, engine="openpyxl")
        except Exception as e:
            st.error(f"Error loading Excel file: {e}")
            return None
    
    elif file_extension == 'json':
        try:
            return pd.read_json(file)
        except Exception as e:
            st.error(f"Error loading JSON file: {e}")
            return None
    
    elif file_extension == 'html':
        try:
            df_list = pd.read_html(file)
            return df_list[0] if df_list else None
        except Exception as e:
            st.error(f"Error loading HTML file: {e}")
            return None
    
    elif file_extension == 'parquet':
        try:
            return pd.read_parquet(file)
        except Exception as e:
            st.error(f"Error loading Parquet file: {e}")
            return None
    
    elif file_extension == 'feather':
        try:
            return pd.read_feather(file)
        except Exception as e:
            st.error(f"Error loading Feather file: {e}")
            return None
    
    elif file_extension == 'pkl':
        try:
            return pd.read_pickle(file)
        except Exception as e:
            st.error(f"Error loading Pickle file: {e}")
            return None
    
    elif file_extension == 'dta':
        try:
            return pd.read_stata(file)
        except Exception as e:
            st.error(f"Error loading Stata file: {e}")
            return None
    
    elif file_extension == 'sav':
        try:
            return pd.read_spss(file)
        except Exception as e:
            st.error(f"Error loading SPSS file: {e}")
            return None
    
    else:
        st.error("Unsupported file format.")
        return None

def explore_data(df):
    """Perform basic exploratory analysis."""
    missing_values = df.isnull().sum()
    duplicates = df.duplicated().sum()
    summary = df.describe(include='all')
    return missing_values, duplicates, summary

import seaborn as sns
import matplotlib.pyplot as plt

def visualize_data(df):
    """Generate visualizations for numerical and categorical features."""
    st.write("### Data Visualizations")

    # Numeric Columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        st.write("#### Distribution of Numeric Features")
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    # Categorical Columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        st.write("#### Distribution of Categorical Features")
        for col in categorical_cols:
            fig, ax = plt.subplots()
            sns.countplot(x=df[col], ax=ax)
            ax.set_title(f"Distribution of {col}")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # Correlation Heatmap
    if len(numeric_cols) > 1:
        st.write("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)


def ask_gemini(missing_values, duplicates, summary):
    """Use Gemini 1.5 Flash to generate structured preprocessing recommendations."""
    model = genai.GenerativeModel("gemini-1.5-flash")

    structured_prompt = f"""
    Given the following dataset insights:
    - Missing Values: {missing_values.to_dict()}
    - Duplicate Rows: {duplicates}
    - Summary Statistics: {summary.to_dict()}

    Provide structured preprocessing recommendations in the following format:

    1) **Columns to Drop** (list columns and justify why)
    2) **New Features to Create** (list new features and why they should be created)
    3) **Preprocessing Steps**:
       - a) **Missing Values Handling**: Which columns have missing values, and how should they be handled? (justify)
       - b) **Duplicate Values Handling**: Which columns may cause duplicate issues? (justify)
       - c) **Outliers Handling**: Which columns have outliers, and how should they be managed? (justify)
       - d) **Feature Encoding**: Which categorical columns need encoding, and which method should be used? (justify)
       - e) **Feature Scaling**: Which numerical columns require scaling, and which technique should be used? (justify)

    4) **Order of Operations**: List the sequence in which these preprocessing steps should be performed.
    5) **suggestions for model building**:Tell which machine learning or deep learning or time series model should be used for the given data and explain why the particular model is good for the given data
    """

    response = model.generate_content(structured_prompt)

    # Store embedding in FAISS
    embedding = np.random.rand(512).astype('float32')  # Placeholder embedding
    st.session_state.faiss_index.add(np.array([embedding]))
    st.session_state.embeddings.append(response.text)

    return response.text



def main():
    st.title("AI Preprocessing and visualization")
    st.sidebar.header("Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx","json","html","parquet","feather","pkl","dta","sav"])
    
    if uploaded_file:
        st.session_state.data = load_data(uploaded_file)
        
        if st.session_state.data is not None:
            df = st.session_state.data.copy()
            st.write("### Data Preview")
            st.dataframe(df.head())

            st.write("### Step 2: Data Exploration")
            missing_values, duplicates, summary = explore_data(df)
            st.write("#### Missing Values", missing_values)
            st.write(f"#### Duplicate Rows: {duplicates} duplicates found")
            st.write("#### Data Summary", summary)

            # Call the visualization function
            visualize_data(df)
            
            st.write("### Step 3: AI Recommendations")
            if st.button("Generate AI Recommendations"):
                response_text = ask_gemini(missing_values, duplicates, summary)
                st.session_state.ai_history.append(response_text)
                st.write(response_text)

            if st.session_state.ai_history:
                st.write("### Previous AI Recommendations")
                for idx, recommendation in enumerate(st.session_state.ai_history[::-1]):  # Show latest first
                    with st.expander(f"Recommendation {len(st.session_state.ai_history) - idx}"):
                        st.write(recommendation)
                # Create downloadable content
                recommendations_text = "\n\n".join(
                    [f"Recommendation {i+1}:\n{rec}" for i, rec in enumerate(st.session_state.ai_history)]
                )
                st.download_button(
                    label="Download AI Recommendations",
                    data=recommendations_text,
                    file_name="ai_recommendations.txt",
                    mime="text/plain"
                )
          

if __name__ == "__main__":
    main()
