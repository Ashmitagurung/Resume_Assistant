# src/utils/resume_info.py
# Resume information extraction utilities

import streamlit as st
from collections import defaultdict


def get_resume_by_role(vectorstore, role):
    """Get all resumes matching a specific role"""
    try:
        # Search for documents with the specified role
        docs = vectorstore.similarity_search(
            query=f"role {role}",
            k=50,  # Get more documents to ensure we capture all relevant ones
            filter={"role": role} if hasattr(vectorstore, 'filter') else None
        )

        # If filter doesn't work, manually filter
        filtered_docs = [doc for doc in docs if doc.metadata.get('role') == role]

        if not filtered_docs:
            # Fallback: search without filter and manually filter
            all_docs = vectorstore.similarity_search(query=role, k=100)
            filtered_docs = [doc for doc in all_docs if doc.metadata.get('role') == role]

        # Group by filename
        resumes = defaultdict(lambda: {'role': role, 'content': []})

        for doc in filtered_docs:
            filename = doc.metadata.get('filename', 'unknown')
            resumes[filename]['content'].append(doc.page_content)

        return dict(resumes)

    except Exception as e:
        st.error(f"Error retrieving resumes by role: {str(e)}")
        return {}


def extract_resume_info(vectorstore, filename):
    """Extract detailed information from a specific resume"""
    try:
        # Search for documents from the specified file
        docs = vectorstore.similarity_search(
            query=f"filename {filename}",
            k=20
        )

        # Filter documents by filename
        file_docs = [doc for doc in docs if doc.metadata.get('filename') == filename]

        if not file_docs:
            return None

        # Combine all content from the file
        full_content = " ".join([doc.page_content for doc in file_docs])

        # Extract metadata
        role = file_docs[0].metadata.get('role', 'Unknown Role')

        return {
            'filename': filename,
            'role': role,
            'content': full_content,
            'chunks': len(file_docs)
        }

    except Exception as e:
        st.error(f"Error extracting resume info: {str(e)}")
        return None


def get_all_filenames(vectorstore):
    """Get all unique filenames from the vector store"""
    try:
        # Get a sample of documents to extract filenames
        docs = vectorstore.similarity_search(query="resume", k=100)

        filenames = set()
        for doc in docs:
            filename = doc.metadata.get('filename', 'unknown')
            if filename != 'unknown':
                filenames.add(filename)

        return list(filenames)

    except Exception as e:
        st.error(f"Error getting filenames: {str(e)}")
        return []