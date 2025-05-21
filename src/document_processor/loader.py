# Document processor: loader.py
# Responsible for loading and processing PDF documents

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os

def process_pdfs(directory_path):
    """Load and process PDFs from a directory with improved metadata extraction"""
    try:
        # Using DirectoryLoader to load all PDFs from the directory
        loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        st.success(f"Successfully loaded {len(documents)} pages from all PDFs")

        if not documents:
            st.warning("No documents were loaded. Please check that valid PDF files were uploaded.")
            return None

        # Add metadata to track which document each chunk came from
        for doc in documents:
            # Extract filename from source path
            filename = os.path.basename(doc.metadata.get('source', 'unknown'))
            doc.metadata['filename'] = filename

            # Try to extract role from content or filename (improved approach)
            content_lower = doc.page_content.lower()

            # Define role mapping with variations
            role_keywords = {
                'AI Engineer': ['ai engineer', 'artificial intelligence engineer', 'machine learning engineer',
                                'ml engineer'],
                'Geomatics Engineer': ['geomatics engineer', 'geomatics', 'geospatial engineer', 'geodetic engineer'],
                'Data Scientist': ['data scientist', 'data science', 'analytics scientist', 'data analyst'],
                'Software Engineer': ['software engineer', 'software developer', 'programmer', 'developer'],
                'UI/UX Designer': ['ui designer', 'ux designer', 'ui/ux', 'user interface', 'user experience',
                                   'product designer'],
                'Project Manager': ['project manager', 'program manager', 'product manager', 'scrum master'],
                'DevOps Engineer': ['devops', 'devops engineer', 'site reliability engineer', 'sre'],
                'Business Analyst': ['business analyst', 'business analytics', 'systems analyst'],
                'Network Engineer': ['network engineer', 'network administrator', 'network architect']
            }

            assigned_role = 'Unknown Role'

            # Check both content and filename for role keywords
            for role, keywords in role_keywords.items():
                if any(keyword in content_lower for keyword in keywords) or any(
                        keyword in filename.lower() for keyword in keywords):
                    assigned_role = role
                    break

            doc.metadata['role'] = assigned_role

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        st.success(f"Split documents into {len(chunks)} chunks")

        return chunks

    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return None