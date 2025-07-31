import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# --- App Configuration ---
st.set_page_config(
    page_title="DGMS Mines Rules Chatbot",
    page_icon="⛏️",
    layout="wide"
)

# --- Caching Functions for Performance ---

@st.cache_resource
def load_and_process_docs():
    """
    Loads PDF documents from the 'docs' directory and processes them.
    This function is cached to avoid reloading and splitting on every run.
    """
    # Ensure the 'docs' directory exists
    if not os.path.exists("docs"):
        st.error("The 'docs' directory is missing. Please create it and add your DGMS PDF files.")
        st.stop()
        
    # Load documents from the specified directory
    loader = PyPDFDirectoryLoader("docs")
    documents = loader.load()
    if not documents:
        st.error("No PDF documents found in the 'docs' directory. Please add your DGMS files.")
        st.stop()

    # Split the documents into smaller, manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    return docs

@st.cache_resource
def create_vector_store(_docs): # The underscore indicates the input is used for caching but not directly in the function body
    """
    Creates and saves a FAISS vector store from the document chunks.
    This function is cached to avoid re-creating the index on every run.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(_docs, embedding=embeddings)
    return vector_store

# --- Main App Logic ---

st.title("⛏️ DGMS Mines Rules & Regulations Chatbot")
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .stTextInput > div > div > input {
            background-color: #ffffff;
        }
    </style>
    """, unsafe_allow_html=True)

st.info("Ask any question about the DGMS rules and regulations based on the provided documents. The chatbot will retrieve the relevant information and generate a detailed answer.")

# Load Groq API Key from Streamlit secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY is not set in Streamlit secrets. Please add it to proceed.")
    st.stop()

# Load and process documents
with st.spinner("Loading and processing documents... This may take a moment on first run."):
    docs = load_and_process_docs()
    vector_store = create_vector_store(docs)

# Initialize the Groq LLM
llm = ChatGroq(
    api_key=groq_api_key,
    model="llama3-8b-8192"
)

# Define the prompt template for the RAG chain
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert assistant for answering questions based on the DGMS Mines Rules and Regulations.
    Answer the user's question strictly based on the context provided below.
    - If the information is not in the context, clearly state that the answer is not available in the provided documents.
    - Provide a detailed and well-structured answer.
    - For each piece of information, cite the source document and page number.

    Context:
    {context}

    Question: {input}

    Answer:
    """
)

# Create the chains for document processing and retrieval
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 relevant chunks
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User input
user_query = st.text_input("Ask your question:", placeholder="e.g., What are the duties of a Welfare Officer?")

if user_query:
    with st.spinner("Searching for the answer..."):
        response = retrieval_chain.invoke({"input": user_query})
        
        # Display the answer
        st.subheader("Answer")
        st.write(response['answer'])
        
        # Display the sources in an expander
        with st.expander("Show Sources"):
            st.subheader("Sources Used to Generate the Answer:")
            unique_sources = {}
            for doc in response['context']:
                source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page = doc.metadata.get('page', 'N/A')
                if page != 'N/A':
                    page += 1 # Page numbers are 0-indexed
                
                # Use a tuple of (source, page) as the key to handle uniqueness
                source_key = (source, page)
                if source_key not in unique_sources:
                    unique_sources[source_key] = doc.page_content

            if unique_sources:
                for (source, page), content in unique_sources.items():
                    st.markdown(f"**📄 Document:** `{source}`  **Page:** `{page}`")
                    st.text_area(label="", value=content, height=150, key=f"source_{source}_{page}")
            else:
                st.write("No specific source documents were retrieved for this query.")
