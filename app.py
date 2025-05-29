# Resume Assistant System
# Main Streamlit application file

import streamlit as st
import os
import tempfile

from src.document_processor.loader import process_pdfs
from src.embeddings.model import initialize_embeddings
from src.retrieval.vectorstore import create_vector_store, get_all_roles
from src.retrieval.qa_chain import initialize_llm, setup_retrieval_system, setup_modern_retrieval_system
from src.utils.resume_info import get_resume_by_role, extract_resume_info
from src.interface.tabs import render_query_tab, render_role_tab, render_modification_tab

# Set page configuration
st.set_page_config(
    page_title="Multi-Resume Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("📄 Multi-Resume Assistant")
st.markdown("""
This app helps you analyze and manage multiple resumes. Upload PDF resumes, ask questions about them,
find candidates with specific skills, and get suggestions for resume improvements.
""")

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'resume_dir' not in st.session_state:
    st.session_state.resume_dir = tempfile.mkdtemp()
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'all_roles' not in st.session_state:
    st.session_state.all_roles = []
if 'chain_type' not in st.session_state:
    st.session_state.chain_type = "traditional"  # or "modern"


# Function to query the RAG system
def query_rag(qa_chain, question, chain_type="traditional"):
    """Enhanced RAG query function with support for different chain types."""
    with st.spinner("Analyzing resumes..."):
        try:
            if chain_type == "modern":
                # For modern retrieval chain
                result = qa_chain.invoke({"input": question})
                return {
                    "result": result.get("answer", "No answer found"),
                    "source_documents": result.get("context", [])
                }
            else:
                # For traditional RetrievalQA chain
                result = qa_chain.invoke({"query": question})
                return result

        except Exception as e:
            # Try alternative input key
            try:
                if chain_type == "traditional":
                    # Try with 'question' key instead of 'query'
                    result = qa_chain.invoke({"question": question})
                    return result
                else:
                    # Try with 'query' key for modern chain
                    result = qa_chain.invoke({"query": question})
                    return {
                        "result": result.get("answer", "No answer found"),
                        "source_documents": result.get("context", [])
                    }
            except Exception as e2:
                st.error(f"Error querying system: {str(e)} | {str(e2)}")
                return {"result": "Error occurred during query", "source_documents": []}


# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # Check if Groq API key is available in secrets
    if "groq" not in st.secrets or "api_key" not in st.secrets["groq"]:
        st.error("❌ Groq API key not found in secrets. Please add it to .streamlit/secrets.toml")
        st.code("""
# Example secrets.toml structure:
[groq]
api_key = "your_groq_api_key_here"
        """)
    else:
        # Display API status
        st.success("✅ Groq API key loaded from secrets")

    # Chain type selection
    st.markdown("### Chain Configuration")
    chain_option = st.selectbox(
        "Select QA Chain Type:",
        ["Traditional RetrievalQA", "Modern Retrieval Chain"],
        help="Choose between traditional or modern LangChain patterns"
    )

    st.session_state.chain_type = "modern" if "Modern" in chain_option else "traditional"

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Resume PDFs",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF resumes to analyze"
    )

    # Process button
    process_button = st.button("Process Resumes", disabled=not uploaded_files)

    # Info section
    st.markdown("---")
    st.markdown("### Usage Tips")
    st.markdown("""
    1. Upload PDF resumes
    2. Choose chain type
    3. Process the resumes
    4. Ask questions or use the tools
    """)

# Process files when button is clicked
if process_button and uploaded_files:
    with st.spinner("Processing uploaded resumes..."):
        # Clear previous files
        for file_path in st.session_state.processed_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        st.session_state.processed_files = []

        # Save uploaded files to the temp directory
        for uploaded_file in uploaded_files:
            file_path = os.path.join(st.session_state.resume_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.processed_files.append(file_path)

        st.success(f"Saved {len(uploaded_files)} resumes to temporary directory")

        try:
            # Process PDFs
            chunks = process_pdfs(st.session_state.resume_dir)
            if not chunks:
                st.error("Failed to process PDFs. Please try again.")
            else:
                # Initialize embeddings
                embeddings = initialize_embeddings()
                if not embeddings:
                    st.error("Failed to initialize embeddings. Please try again.")
                else:
                    # Create vector store
                    vectorstore = create_vector_store(chunks, embeddings)
                    if not vectorstore:
                        st.error("Failed to create vector store. Please try again.")
                    else:
                        # Initialize LLM
                        llm = initialize_llm()
                        if not llm:
                            st.error("Failed to initialize LLM. Please check your API key in the secrets.toml file.")
                        else:
                            # Setup retrieval system based on selected type
                            if st.session_state.chain_type == "modern":
                                qa_chain = setup_modern_retrieval_system(llm, vectorstore)
                            else:
                                qa_chain = setup_retrieval_system(llm, vectorstore)

                            if not qa_chain:
                                st.error("Failed to set up retrieval system. Please try again.")
                            else:
                                # Get all roles
                                all_roles = get_all_roles(vectorstore)

                                # Save to session state
                                st.session_state.vectorstore = vectorstore
                                st.session_state.qa_chain = qa_chain
                                st.session_state.all_roles = all_roles
                                st.session_state.initialized = True

                                st.success(f"✅ System initialized successfully with {chain_option}!")

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.error("Please check your module imports and file structure.")

# Main content area - split into tabs
if st.session_state.initialized:
    tab1, tab2, tab3 = st.tabs(["Query Resumes", "Search by Role", "Resume Modification"])

    # Tab 1: Free-form querying
    with tab1:
        render_query_tab(
            st.session_state.initialized,
            st.session_state.qa_chain,
            lambda qa_chain, question: query_rag(qa_chain, question, st.session_state.chain_type)
        )

    # Tab 2: Role-based search
    with tab2:
        render_role_tab(
            st.session_state.initialized,
            st.session_state.all_roles,
            st.session_state.vectorstore,
            st.session_state.qa_chain,
            get_resume_by_role,
            lambda qa_chain, question: query_rag(qa_chain, question, st.session_state.chain_type)
        )

    # Tab 3: Resume modification suggestions
    with tab3:
        render_modification_tab(
            st.session_state.initialized,
            st.session_state.all_roles,
            st.session_state.qa_chain,
            st.session_state.vectorstore,
            get_resume_by_role,
            lambda qa_chain, question: query_rag(qa_chain, question, st.session_state.chain_type)
        )
else:
    # Display initialization status
    st.warning("Please upload PDF resumes and process them to begin.")

    # Show some information about the system
    st.markdown("""
 
    """)