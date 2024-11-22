import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from typing_extensions import TypedDict
from typing import Literal
import cassio
import requests
from io import BytesIO
import os
from dotenv import load_dotenv


# RouteQuery model for structured output
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        ...,
        description="Given a user question choose to route it to wikipedia or a vectorstore.",
    )

# Load environment variables from .env file
load_dotenv()

# Access keys
ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize the ASTRA DB connection
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")
structured_llm_router = llm.with_structured_output(RouteQuery)

# Initialize vector store (Cassandra)
vector_store = Cassandra(embedding=embeddings, table_name="demo_qa", session=None, keyspace=None)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
retriever = vector_store.as_retriever()

# Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Add fallback for None pages
    return text


# Function to extract text from PDFs in a URL
def extract_text_from_url(urls):
    # Fix any invalid URLs (missing scheme)
    def validate_and_fix_url(url):
        # Check if the URL has a scheme (http:// or https://)
        if not url.startswith(('http://', 'https://')):
            # If no scheme is provided, add https:// by default
            return 'https://' + url
        return url

    # Ensure all URLs have valid schemes
    valid_urls = [validate_and_fix_url(url) for url in urls]
    
    # Load documents from valid URLs
    docs = [WebBaseLoader(url).load() for url in valid_urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    return docs_list


# Function to process documents and add them to the vector store
def process_documents(documents):
    split_documents = text_splitter.split_documents(documents)
    vector_store.add_documents(split_documents)
    return vector_store.as_retriever()


# Workflow functions
def retrieve(state):
    """
    Retrieve documents from the vector store.
    
    Args:
        state (dict): The current graph state
    
    Returns:
        state (dict): Updated state with retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def route_question(state):
    """
    Route the user query to either wiki search or vector store.
    
    Args:
        state (dict): The current graph state
    
    Returns:
        str: Next node to call based on datasource (either "wiki_search" or "vectorstore")
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    
    if source.datasource == "wiki_search":
        print("---ROUTE QUESTION TO Wiki SEARCH---")
        return "wiki_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def wiki_search(state):
    """
    Perform a Wikipedia search based on the re-phrased question.
    
    Args:
        state (dict): The current graph state
    
    Returns:
        state (dict): Updates the state with retrieved web results from Wikipedia
    """
    print("---wikipedia---")
    question = state["question"]
    docs = wiki.invoke({"query": question})
    wiki_results = docs
    wiki_results = Document(page_content=wiki_results)
    return {"documents": wiki_results, "question": question}


# Prompt for routing user queries
system = """You are an expert at routing a user question to a vectorstore or wikipedia.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use wiki-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router


# Graph state definition for the workflow
class GraphState(TypedDict):
    """Represents the state of our graph."""
    question: str
    generation: str
    documents: List[str]


# Define the workflow graph
workflow = StateGraph(GraphState)

# Add nodes to the workflow graph
workflow.add_node("wiki_search", wiki_search)  # Web search node
workflow.add_node("retrieve", retrieve)  # Document retrieval node

# Add conditional edges based on the question routing
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "wiki_search": "wiki_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", END)
workflow.add_edge("wiki_search", END)

# Compile the workflow graph
app = workflow.compile()


# Streamlit UI setup
st.title("Chatbot with PDF(s) or URL(s) Data")

# Option 1: Upload multiple PDF files
uploaded_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

pdf_texts = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        
        text = extract_text_from_pdf(uploaded_file)
        if text:
            pdf_texts.append(text)

# Option 2: Provide multiple URLs of PDFs
urls = st.text_area("Or, provide URLs of PDF(s), one per line").splitlines()

# Extract text from each URL and flatten the result
for url in urls:
    if url.strip():
        docs_list = extract_text_from_url([url.strip()])
        if docs_list:
            # Flatten list of documents (each doc is a list of page_content)
            pdf_texts.extend([doc.page_content for doc in docs_list])

# Combining all the extracted text
full_text = "\n".join(pdf_texts)

# Process and store documents in the vector store
if full_text:
    # Process the combined text into the vector store
    split_documents = text_splitter.split_documents([Document(page_content=full_text)])
    vector_store.add_documents(split_documents)
    
    # Start the workflow and prompt the user to ask a question
    st.subheader("Ask a Question:")

    user_query = st.text_input("Enter your question:")
    if user_query:
        with st.spinner("Generating response..."):
            # Run the workflow (graph processing) to answer the question
            state = {"question": user_query}
            workflow_result = app.invoke(state)

            # Get the response based on the retrieved documents and answer generation
            response = workflow_result["documents"]

            # Show the chatbot response
            st.subheader("Chatbot Response:")
            st.write(response)

else:
    st.warning("No data available for the chatbot. Please upload PDFs or provide URLs.")
