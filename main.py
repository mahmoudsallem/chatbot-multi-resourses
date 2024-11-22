import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.document_loaders.pdf import PDFMinerLoader
from langchain_community.vectorstores import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
import requests
import os
import cassio
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access keys
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
groq_api_key = os.getenv('GROQ_API_KEY')

## connection of the ASTRA DB
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)


# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

from langchain.vectorstores.cassandra import Cassandra

vector_store=Cassandra(embedding=embeddings,
                              table_name="demo_qa",
                              session=None,
                              keyspace=None,)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to load documents
def load_documents(url):
    if url.endswith(".pdf"):
        loader = PDFMinerLoader(url)
    else:
        loader = WebBaseLoader(url)
    return loader.load()

# Function to process documents
def process_documents(documents):
    split_documents = text_splitter.split_documents(documents)
    vector_store.add_documents(split_documents)
    return vector_store.as_retriever()

st.title("PDF Upload or URL Input")

# Option 1: Upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Read the uploaded PDF
    text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Text from Uploaded PDF:")
    st.text_area("PDF Content", text, height=300)

# Option 2: Provide a URL of a PDF
url = st.text_input("Or, provide a URL of a PDF")

if url:
    try:
        # Get the PDF file from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        # Extract text from the PDF at the URL
        text = extract_text_from_pdf(io.BytesIO(response.content))
        st.subheader("Extracted Text from PDF URL:")
        st.text_area("PDF Content", text, height=300)

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching PDF from URL: {e}")
