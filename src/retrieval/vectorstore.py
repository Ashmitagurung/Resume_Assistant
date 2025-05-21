# Vector store creation and management
# Responsible for creating and interacting with the vector store

import os
import shutil
import time
import streamlit as st
from langchain_community.vectorstores import Chroma


def create_vector_store(chunks, embeddings):
    """Create embeddings and vector store from document chunks with validation"""
    try:
        if not chunks:
            raise ValueError("No document chunks available to create vector store")

        persist_directory = "./data/chroma_db"
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(persist_directory), exist_ok=True)

        # Remove existing DB if it exists to avoid conflicts
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        with st.spinner(f"Creating vector store with {len(chunks)} chunks..."):
            start_time = time.time()

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )

            vectorstore.persist()
            creation_time = time.time() - start_time
            st.success(f"Vector store created and persisted in {creation_time:.2f} seconds")

        # Validation - check if embeddings can be retrieved
        with st.spinner("Validating vector store..."):
            test_query = "skills"
            test_results = vectorstore.similarity_search(test_query, k=2)

            if test_results:
                st.success(f"Successfully validated vector store with {len(test_results)} test results")
            else:
                st.warning("Validation found no results in test query")

        return vectorstore

    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


def get_all_roles(vectorstore):
    """Get all unique roles from the vector store."""
    try:
        # Query the vector store to get a sample of documents
        all_docs = vectorstore.similarity_search(
            query="all resumes",
            k=100  # Get a large sample to ensure we capture all roles
        )

        roles = set()
        for doc in all_docs:
            role = doc.metadata.get('role', 'Unknown Role')
            roles.add(role)

        return list(roles)
    except Exception as e:
        st.error(f"Error getting roles: {str(e)}")
        return []