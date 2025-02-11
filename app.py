import re, os
import pickle
import joblib
import pdfminer
import numpy as np
import pandas as pd

import streamlit as st
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from pdfminer.high_level import extract_pages
from sklearn.metrics.pairwise import cosine_similarity

unified_list = [
    'web', 'design', 'website', 'development', 'application', 'service', 'management', 'developer', 'designer', 'software', 
    'developed', 'site', 'marketing', 'business', 'html', 'content', 'state', 'cs', 'javascript', 'data', 'created', 'graphic', 
    'city', 'system', 'client', 'testing', 'test', 'patient', 'care', 'plan', 'procedure', 'information', 'student', 'tool', 
    'security', 'professional', 'staff', 'program', 'equipment', 'medical', 'sale', 'product', 'need', 'relationship', 'account', 
    'maintained', 'store', 'strategy', 'merchandise', 'communication', 'goal', 'revenue', 'inventory', 'market', 'area', 'current', 
    'sql', 'database', 'server', 'ssis', 'query', 'package', 'table', 'oracle', 'production', 'stored', 'solution', 'ssrs', 'tuning', 
    'implementation', 'abap', 'consultant', 'analysis', 'module', 'master', 'integration', 'configuration', 'react', 'spring', 
    'code', 'java', 'implemented', 'framework', 'python', 'aws', 'django', 'medium', 'public', 'relation', 'event', 'social', 
    'press', 'including', 'pr', 'community', 'campaign', 'pmo', 'operation', 'budget', 'resource', 'improvement', 'risk', 'change', 
    'leadership', 'financial', 'agile', 'employee', 'policy', 'daily', 'safety', 'operational', 'cost', 'member', 'network', 
    'cisco', 'firewall', 'engineer', 'switch', 'troubleshooting', 'router', 'infrastructure', 'access', 'hardware', 'device', 
    'routing', 'mechanical', 'engineering', 'manufacturing', 'maintenance', 'component', 'material', 'machine', 'part', 'solidworks', 
    'designed', 'specification', 'record', 'technology', 'computer', 'problem', 'hr', 'human', 'benefit', 'payroll', 'recruitment', 
    'compensation', 'department', 'compliance', 'fitness', 'health', 'exercise', 'group', 'personal', 'class', 'wellness', 'nutrition', 
    'trainer', 'certified', 'food', 'beverage', 'guest', 'restaurant', 'menu', 'standard', 'item', 'hotel', 'order', 'satisfaction', 
    'finance', 'accounting', 'tax', 'reconciliation', 'statement', 'prepared', 'audit', 'entry', 'payment', 'electrical', 'repair', 
    'wiring', 'circuit', 'installed', 'electrician', 'teacher', 'learning', 'classroom', 'lesson', 'teaching', 'educational', 
    'instruction', 'curriculum', 'positive', 'child', 'parent', 'behavior', 'etl', 'informatica', 'source', 'mapping', 'warehouse', 
    'file', 'net', 'aspnet', 'c', 'jquery', 'digital', 'brand', 'creative', 'google', 'devops', 'cloud', 'deployment', 'jenkins', 
    'docker', 'continuous', 'linux', 'azure', 'pipeline', 'git', 'automation', 'art', 'adobe', 'interior', 'recovery', 'administrator', 
    'monitoring', 'microsoft', 'science', 'model', 'research', 'year', 'scientist', 'civil', 'construction', 'building', 'water', 
    'analyst', 'stakeholder', 'use', 'diagram', 'worker', 'contractor', 'schedule', 'blockchain', 'faculty', 'contract', 'smart', 
    'york', 'banking', 'branch', 'loan', 'credit', 'cash', 'call', 'center', 'agent', 'aircraft', 'aviation', 'flight', 'pilot', 
    'personnel', 'vehicle', 'automotive', 'claim', 'director', 'english', 'language', 'grade', 'architect', 'architecture', 
    'architectural', 'planning', 'apparel', 'associate', 'display', 'organized', 'floor', 'merchandising', 'fashion', 'agricultural', 
    'farm', 'crop', 'agriculture', 'plant', 'advocate', 'legal', 'victim', 'family', 'provided'
]

lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')

model_dir = "models"
w2v_preprocessor_path = os.path.join(model_dir, "word2vec_model.model")
w2v_gb_model_path = os.path.join(model_dir, "word2vec_XGBoost_model.pkl")
w2v_label_encoder_path = os.path.join(model_dir, "label_encoder2.pkl")

tfidf_preprocessor_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
tfidf_gb_model_path = os.path.join(model_dir, "tfidf_XGBoost_model.pkl")
tfidf_label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")

def extract_text(data):
    extracted_text = re.findall(r"'(.*?)'", data)

    if len(extracted_text) > 0:
        return extracted_text[0]
    return ""

# Resume text Cleaning
def clean_text(text):
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    links_pattern = re.compile(r'https?:\/\/\S+|www\.\S+')
    cleaned_text = text.replace(r'\n', ' ').replace('\n', ' ')

    cleaned_text = email_pattern.sub('',cleaned_text)
    cleaned_text = links_pattern.sub('',cleaned_text)

    cleaned_text = re.sub('[^\w\s]','',cleaned_text)
    cleaned_text = re.sub(r'[â]', '', cleaned_text) 
    cleaned_text = re.sub(r'[ã]', '', cleaned_text) 
    cleaned_text = re.sub(r'\b[1-9][0-9]?\b', '', cleaned_text)
    
    cleaned_text = cleaned_text.lower()

    cleaned_text = " ".join([word for word in cleaned_text.split() if word not in stopwords])
    cleaned_text = " ".join([word for word in cleaned_text.split() if word in unified_list])
    cleaned_text = " ".join([lemmatizer.lemmatize(word) for word in cleaned_text.split()])

    return cleaned_text

def compute_similarity(job_desc, resume_text):
    """Compute relevance score using TF-IDF and cosine similarity."""
    vectorizer = joblib.load(tfidf_preprocessor_path)
    
    # Combine both texts for vectorization
    combined_texts = [job_desc, resume_text]
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    
    # Calculate cosine similarity and return the score out of 100
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(similarity_score * 100, 2)

def transform_text_data_w2v(cleaned_data):
    if os.path.exists(w2v_preprocessor_path):
        word2vec_model = Word2Vec.load(w2v_preprocessor_path)

        vectors = [word2vec_model.wv[word] for word in cleaned_data.split() if word in word2vec_model.wv]

        return np.mean(vectors,axis=0).reshape(1,-1)
    else:
        print("word2vec vectorizer does not exists")

    return np.zeros((1, word2vec_model.vector_size))

def predict_result_w2v(features):
    if os.path.exists(w2v_label_encoder_path):
        with open(w2v_label_encoder_path, 'rb') as file:
            le = pickle.load(file)
        labelled_data = {value: name for value, name in enumerate(le.classes_)}

        if os.path.exists(w2v_gb_model_path):
            with open(w2v_gb_model_path, 'rb') as file:
                model_GB = pickle.load(file)
            
            num_features = model_GB.n_features_in_
            feature_columns = [f"feature_{i}" for i in range(num_features)]
            features_df = pd.DataFrame(features, columns=feature_columns)

            y_pred = model_GB.predict(features_df)

            print(y_pred)
            print([labelled_data[pred] for pred in y_pred])
            # return labelled_data[y_pred]
            return [labelled_data[pred] for pred in y_pred]
        else:
            print("word2vec_GB Model file does not exist.")
    else:
        print("Label Ecoder Path does not exists")
        
    return ["error"]

def transform_text_data_tfidf(cleaned_data):
    if os.path.exists(tfidf_preprocessor_path):
        tfidf = joblib.load(tfidf_preprocessor_path)
        tfidf_vector = tfidf.transform([cleaned_data])
        return tfidf_vector.toarray()
    else:
        print("tfidf vectorizer does not exists")

    return [0]

def predict_result_tfidf(features):

    if os.path.exists(tfidf_label_encoder_path):
        with open(tfidf_label_encoder_path, 'rb') as file:
            le = pickle.load(file)
        labelled_data = {value: name for value, name in enumerate(le.classes_)}

        if os.path.exists(tfidf_gb_model_path):
            with open(tfidf_gb_model_path, 'rb') as file:
                model_GB = pickle.load(file)
            

            y_pred = model_GB.predict(features)

            print(y_pred)
            print([labelled_data[pred] for pred in y_pred])
            # return labelled_data[y_pred]
            return [labelled_data[pred] for pred in y_pred]
        else:
            print("tfidf Model file does not exist.")
    else:
        print("Label Ecoder Path does not exists")
        
    return ["error"]


 
if __name__ == "__main__":
    text_data = []
    uploaded_file = st.file_uploader("Upload your CV (PDF)", "pdf")
    if uploaded_file is not None:
        for page_layout in extract_pages(uploaded_file):
            for element in page_layout:
                data = extract_text(str(element))
                if len(data) != 0:
                    text_data.append(data)
        
        final_text = " ".join(text_data)
        print(final_text)
        cleaned_data = clean_text(final_text)
        print(cleaned_data)

        file_path = "data/new_uploaded_cvs/extracted_cv_text.txt"
        folder_path = os.path.dirname(file_path)

        # Check if the folder exists; create it if not
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(file_path, "a", encoding="utf-8") as text_file:
            text_file.write(cleaned_data)
            text_file.write("\n")  # Add a newline before writing the new text

        st.success("Text saved successfully!")

        features1 = transform_text_data_w2v(cleaned_data)
        result1 = predict_result_w2v(features1)

        features2 = transform_text_data_tfidf(cleaned_data)
        result2 = predict_result_tfidf(features2)

        st.write(f'word2vec Model: The CV is of {result1[0]}')
        st.write(f'Tf-Idf Model: The CV is of {result2[0]}')
            

        job_desc = st.text_input("Enter job description data")

        clean_job_desc = clean_text(job_desc)
        clean_resume_text = clean_text(cleaned_data)
        score = compute_similarity(clean_job_desc, clean_resume_text)
        
        st.success(f"Relevance Score: {score}/100")

        