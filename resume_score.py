import re
import spacy
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

skill_list = {"python", "sql", "aws", "machine learning", "deep learning", "nlp", "data analysis", "tensorflow", "pytorch", "java", "c++", "kubernetes"}

def extract_skills(text):
    doc = nlp(text.lower())
    extracted_skills = {token.text for token in doc if token.text in skill_list}
    return extracted_skills

def extract_experience(text):
    match = re.findall(r"(\d+)\s*(?:years|yrs|year|yr)", text.lower())
    if match:
        return min(15, int(max(match)))  # Cap at 15 years
    return 0

def extract_education(text):
    degrees = {"phd": 40, "mba": 30, "master": 25, "bachelor": 20, "diploma": 10}
    for degree, score in degrees.items():
        if degree in text.lower():
            return score
    return 5  # Default minimum score

def compute_text_similarity(resume_text, job_text):
    embeddings = bert_model.encode([resume_text, job_text], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return similarity_score

def compute_resume_score(resume_text, job_description):
    resume_skills = extract_skills(resume_text)
    experience = extract_experience(resume_text)
    education = extract_education(resume_text)
    job_skills = extract_skills(job_description)
    
    skill_match = len(resume_skills.intersection(job_skills)) / max(len(job_skills), 1) * 100
    text_similarity = compute_text_similarity(resume_text, job_description) * 100
    
    final_score = (0.4 * skill_match) + (0.3 * experience) + (0.2 * education) + (0.1 * text_similarity)
    
    return round(final_score, 2)

# Example Usage
resumes = [
    {"id": 1, "text": "Python developer with 8 years experience in AWS, Kubernetes, and SQL. Holds a Master's degree."},
    {"id": 2, "text": "Data scientist with expertise in NLP, deep learning, and 4 years of experience. Holds a PhD."}
]

job_description = "Looking for a Python developer with AWS, Kubernetes, and SQL skills."

ranked_resumes = []
for resume in resumes:
    score = compute_resume_score(resume["text"], job_description)
    ranked_resumes.append((resume["id"], score))

# Sort resumes by score
display_ranked_resumes = sorted(ranked_resumes, key=lambda x: x[1], reverse=True)
print(display_ranked_resumes)
