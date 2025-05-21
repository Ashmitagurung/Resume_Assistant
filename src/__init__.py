
# Initialize package files
# Create empty __init__.py files for all subdirectories

"""src/__init__.py: Main package for Resume Assistant"""

__version__ = '0.1.0'
__author__ = 'Ashmit'

# Import main components for easier access
from src.document_processor.loader import process_pdfs
from src.embeddings.model import initialize_embeddings
from src.retrieval.vectorstore import create_vector_store, get_all_roles
from src.retrieval.qa_chain import initialize_llm, setup_retrieval_system
from src.utils.resume_info import get_resume_by_role, extract_resume_info