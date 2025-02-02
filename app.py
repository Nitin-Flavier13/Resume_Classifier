import re, os
import pdfminer
import streamlit as st
from pdfminer.high_level import extract_pages

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stopwords = stopwords.words('english')

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
    cleaned_text = " ".join([lemmatizer.lemmatize(word) for word in cleaned_text.split()])

    return cleaned_text


st.write(pdfminer.__version__)  

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
            
# import os, io
# import docx
# import fitz  # PyMuPDF for PDF extraction
# import streamlit as st

# def extract_text_from_docx(docx_file):
#     doc = docx.Document(docx_file)
#     text = " ".join([para.text for para in doc.paragraphs])
#     return text

# # Function to extract text from the uploaded PDF
# def extract_text_from_pdf(uploaded_file):
#     # Ensure the uploaded file is read into memory as a byte stream
#     pdf_bytes = uploaded_file.read()
    
#     # Open the PDF from the byte stream
#     doc = fitz.open(io.BytesIO(pdf_bytes))  # Open PDF directly from BytesIO object
#     text_content = ""
    
#     # Iterate through the pages and extract text
#     for page in doc:
#         text_content += page.get_text()
    
#     # Close the document after extracting text
#     doc.close()
    
#     return text_content

# def detect_cv_type(text):
#     """Simple heuristic to detect CV layout based on text orientation."""
#     if len(text.split('\n')) > 50:  # Heuristic for vertical CVs (more line breaks)
#         return "Vertical CV"
#     return "Horizontal CV"

# # Streamlit App UI
# st.title("CV Text Extractor")

# uploaded_file = st.file_uploader("Upload your CV (PDF or DOCX)", type=["pdf", "docx"])

# if uploaded_file is not None:
#     file_extension = os.path.splitext(uploaded_file.name)[-1].lower()

#     # Extract text based on file type
#     if file_extension == ".pdf":
#         text_content = extract_text_from_pdf(uploaded_file)
#     elif file_extension == ".docx":
#         text_content = extract_text_from_docx(uploaded_file)
#     else:
#         st.error("Unsupported file type.")
#         st.stop()

#     # Detect CV Type
#     cv_type = detect_cv_type(text_content)

#     st.subheader(f"Detected CV Type: {cv_type}")
#     st.text_area("Extracted CV Content", text_content, height=300)

#     # Save the extracted text for further processing if needed
#     if st.button("Save Extracted Text"):
#         with open("data/new_uploaded_cvs/extracted_cv_text.txt", "w") as text_file:
#             text_file.write(text_content)
#         st.success("Text saved successfully!")