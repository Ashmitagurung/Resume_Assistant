# src/retrieval/qa_chain.py

import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import streamlit as st


def initialize_llm():
    """Initialize the Groq LLM."""
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets["groq"]["api_key"]

        # Initialize ChatGroq
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama3-70b-8192",  # or "mixtral-8x7b-32768"
            temperature=0.1,
            max_tokens=1024
        )

        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None


def setup_retrieval_system(llm, vectorstore):
    """Set up the retrieval QA system."""
    try:
        # Create a custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Method 1: Using RetrievalQA (Original approach - fixed)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=True
        )

        return qa_chain

    except Exception as e:
        st.error(f"Error setting up retrieval system: {str(e)}")
        return None


def setup_modern_retrieval_system(llm, vectorstore):
    """Alternative setup using newer LangChain patterns."""
    try:
        # Create a custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {input}

        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "input"]
        )

        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Create document chain
        document_chain = create_stuff_documents_chain(llm, PROMPT)

        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return retrieval_chain

    except Exception as e:
        st.error(f"Error setting up modern retrieval system: {str(e)}")
        return None