# Embeddings model setup
# Responsible for initializing the embedding model

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings

def initialize_embeddings():
    """Initialize the embedding model"""
    try:
        with st.spinner("Initializing embedding model (BAAI/bge-small-en-v1.5)..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={'device': 'cpu'}  # Use GPU if available
            )
            return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None