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
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar open initially
)

# --- Caching Functions for Performance ---

@st.cache_resource
def load_and_process_docs():
    """
    Loads PDF documents from the 'docs' directory and processes them.
    This function is cached to avoid reloading and splitting on every run.
    """
    if not os.path.exists("docs"):
        st.error("The 'docs' directory is missing. Please create it and add your DGMS PDF files.")
        st.stop()
        
    loader = PyPDFDirectoryLoader("docs")
    documents = loader.load()
    if not documents:
        st.error("No PDF documents found in the 'docs' directory. Please add your DGMS files.")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    return docs

@st.cache_resource
def create_vector_store(_docs):
    """
    Creates and saves a FAISS vector store from the document chunks.
    This function is cached to avoid re-creating the index on every run.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(_docs, embedding=embeddings)
    return vector_store

# --- Main App Logic ---

# Custom CSS for a professional dark theme
st.markdown("""
    <style>
        /* Main app background */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        /* Sidebar styling */
        .st-emotion-cache-163ttbj {
            background-color: #1E1E1E;
        }
        /* Chat input box */
        .st-emotion-cache-16txtl3 {
            background-color: #262730;
            border-radius: 0.5rem;
        }
        /* Sidebar buttons */
        .st-emotion-cache-1v0mbdj > button {
            width: 100%;
            background-color: #262730;
            border: 1px solid #4A4A4A;
            color: #FAFAFA;
            transition: background-color 0.3s ease;
        }
        .st-emotion-cache-1v0mbdj > button:hover {
            background-color: #4A4A4A;
            color: #FFFFFF;
        }
        /* Expander styling */
        .st-emotion-cache-p5msec {
            background-color: #1E1E1E;
            border: 1px solid #4A4A4A;
        }
        /* Source text area */
        .stTextArea textarea {
            background-color: #0E1117;
            color: #D1D1D1;
            border: 1px solid #4A4A4A;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⛏️ DGMS Chatbot")
    st.markdown("""
    This chatbot provides answers based on the official DGMS Mines Rules and Regulations documents.
    
    It uses a **Retrieval-Augmented Generation (RAG)** model to ensure the information is accurate and contextually relevant.
    """)
    
    st.header("Example Questions")
    example_questions = [
        "What are the duties of a Welfare Officer?",
        "What are the rules for first-aid stations?",
        "Explain the precautions against fire in underground mines.",
        "What is the procedure for reporting accidents?"
    ]
    # Use a unique key for each button to avoid conflicts
    for i, question in enumerate(example_questions):
        if st.button(question, key=f"example_{i}"):
            st.session_state.user_query = question
            # Rerun to process the button click immediately
            st.rerun()

# --- Main Content ---
st.title("DGMS Mines Rules & Regulations Assistant")
st.markdown("Your expert assistant for navigating DGMS rules. Ask a question below or select an example from the sidebar.")

# Load Groq API Key from Streamlit secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY is not set in Streamlit secrets. Please add it to proceed.")
    st.stop()

# Load and process documents, showing a spinner
with st.spinner("Initializing the knowledge base... This may take a moment on the first run."):
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
    Your response must be professional, detailed, and strictly based on the context provided.

    **Instructions:**
    1.  Answer the user's question using only the information present in the context below.
    2.  If the context does not contain the answer, state clearly: "The information is not available in the provided documents."
    3.  Structure your answer clearly. Use bullet points or numbered lists if it helps readability.
    4.  At the end of your answer, you MUST cite the sources you used in the format: `[Source: Document Name, Page: Page Number]`.

    **Context:**
    {context}

    **Question:** {input}

    **Answer:**
    """
)

# Create the chains for document processing and retrieval
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input from text input or sidebar buttons
user_input = st.session_state.pop("user_query", None) or st.chat_input("Ask your question about DGMS rules...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input": user_input})
            full_response = response['answer']
            
            st.markdown(full_response)
            
            with st.expander("Show Sources Used"):
                unique_sources = {}
                for doc in response['context']:
                    source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    page = doc.metadata.get('page', 'N/A')
                    if page != 'N/A':
                        page += 1
                    
                    source_key = (source, page)
                    if source_key not in unique_sources:
                        unique_sources[source_key] = doc.page_content

                if unique_sources:
                    for (source, page), content in unique_sources.items():
                        st.markdown(f"**📄 Document:** `{source}`  **Page:** `{page}`")
                        st.text_area(label="", value=content, height=150, key=f"source_{source}_{page}")
                else:
                    st.warning("No specific source documents were retrieved to formulate this answer.")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
