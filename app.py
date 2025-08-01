import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables from .env file
load_dotenv()

# --- App Configuration ---
st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Functions for Performance ---
@st.cache_resource
def create_embeddings_model():
    """Loads the embedding model, cached for performance."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
        [data-testid="stSidebar"] {
            background-color: #1E1E1E;
        }
        /* Chat input box */
        .st-emotion-cache-16txtl3 {
            background-color: #262730;
            border-radius: 0.5rem;
        }
        /* Sidebar buttons */
        [data-testid="stSidebar"] button {
            width: 100%;
            background-color: #262730;
            border: 1px solid #4A4A4A;
            color: #FAFAFA;
            transition: background-color 0.3s ease;
        }
        [data-testid="stSidebar"] button:hover {
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
        /* Feedback buttons */
        .feedback-buttons {
            display: flex;
            gap: 5px;
            margin-top: 10px;
        }
        .feedback-buttons button {
            background-color: #262730;
            border: 1px solid #4A4A4A;
            color: #FAFAFA;
            border-radius: 5px;
            padding: 2px 8px;
            cursor: pointer;
        }
        .feedback-buttons button:hover {
            background-color: #4A4A4A;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# --- Sidebar ---
with st.sidebar:
    st.header("📄 Document Q&A Chatbot")
    st.markdown("""
    Upload your PDF documents, and this chatbot will answer your questions based on their content. 
    It now remembers the context of your conversation.
    """)
    
    uploaded_files = st.file_uploader(
        "Upload your PDF documents",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload one or more PDF files."
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents... This may take a moment."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    all_docs = []
                    for uploaded_file in uploaded_files:
                        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_filepath, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        loader = PyPDFLoader(temp_filepath)
                        all_docs.extend(loader.load())
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                docs_chunks = text_splitter.split_documents(all_docs)
                
                embeddings = create_embeddings_model()
                st.session_state.vector_store = FAISS.from_documents(docs_chunks, embedding=embeddings)
                st.session_state.chat_history = [] # Clear previous chat history
                st.session_state.feedback = {} # Clear feedback
                st.success("Documents processed successfully! You can now ask questions.")
        else:
            st.warning("Please upload at least one PDF document to process.")

# --- Main Content ---
st.title("Your Personal Document Assistant")
st.markdown("Upload your documents in the sidebar and start a conversation.")

# Load Groq API Key from Streamlit secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY is not set in Streamlit secrets. Please add it to proceed.")
    st.stop()

# Display chat messages from history
for i, message in enumerate(st.session_state.chat_history):
    if message.type == "human":
        with st.chat_message("user"):
            st.markdown(message.content)
    elif message.type == "ai":
        with st.chat_message("assistant"):
            st.markdown(message.content)
            # Display feedback buttons for assistant messages
            feedback_key = f"feedback_{i}"
            if feedback_key not in st.session_state.feedback:
                cols = st.columns([1, 1, 10])
                with cols[0]:
                    if st.button("👍", key=f"thumbs_up_{i}"):
                        st.session_state.feedback[feedback_key] = "positive"
                        st.rerun()
                with cols[1]:
                    if st.button("👎", key=f"thumbs_down_{i}"):
                        st.session_state.feedback[feedback_key] = "negative"
                        st.rerun()
            elif st.session_state.feedback[feedback_key] == "positive":
                st.markdown("👍 Thank you for your feedback!")
            elif st.session_state.feedback[feedback_key] == "negative":
                st.markdown("👎 Thank you for your feedback!")

# Handle user input
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process your documents before asking a question.")
    else:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")
                retriever = st.session_state.vector_store.as_retriever()

                # 1. History-aware retriever chain
                contextualize_q_system_prompt = """Given a chat history and the latest user question \
                which might reference context in the chat history, formulate a standalone question \
                which can be understood without the chat history. Do NOT answer the question, \
                just reformulate it if needed and otherwise return it as is."""
                contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, contextualize_q_prompt
                )

                # 2. Question-Answering chain
                qa_system_prompt = """You are an expert assistant for answering questions based on provided documents.
                Your response must be professional, detailed, and strictly based on the context provided.

                **Instructions:**
                1. Answer the user's question using only the information present in the context.
                2. If the context does not contain the answer, state clearly: "The information is not available in the provided documents."
                3. Structure your answer clearly. Use bullet points or numbered lists if it helps readability.
                4. At the end of your answer, you MUST cite the sources you used in the format: `[Source: Document Name, Page: Page Number]`.

                **Context:**
                {context}

                **Question:** {input}

                **Answer:**
                """
                qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", qa_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

                # 3. Combine into the final RAG chain
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                # Invoke the chain with history
                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                
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
                            st.text_area(label="", value=content, height=150, key=f"source_{source}_{page}_{user_input}")
                    else:
                        st.warning("No specific source documents were retrieved to formulate this answer.")

        # Update chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=full_response))
