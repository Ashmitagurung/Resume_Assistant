# QA Chain setup
# Responsible for initializing the language model and setting up the retrieval QA chain

import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

def initialize_llm():
    """Initialize the language model using API key from secrets"""
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets["groq"]["api_key"]

        with st.spinner("Initializing language model..."):
            llm = ChatGroq(
                model_name="Llama-3.3-70b-Versatile",
                temperature=0.2,  # Lower temperature for more factual responses
                max_tokens=1000,  # Increased token limit for more detailed responses
                api_key=api_key,
            )

            # Test the LLM to ensure it works
            test_response = llm.invoke("Give a brief response to test if you're working properly.")
            st.success("LLM connection successful!")

            return llm

    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def setup_retrieval_system(llm, vectorstore):
    """Set up the RAG retrieval system"""
    try:
        with st.spinner("Setting up RAG retrieval system..."):
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}  # Increased to 8 to get more context from multiple documents
            )

            # Create a custom QA prompt template for better instructions
            qa_template = """You are a Resume Analysis Expert assistant.

            You need to answer questions about multiple resumes. Use ONLY the context provided below to answer. If you don't know the answer based on the context, say "I don't have that information in the provided resumes."

            When answering, always:
            1. Specify which resume (by role and filename) you're referring to
            2. Use direct quotes from the resumes when appropriate
            3. Organize information clearly

            Context about the resumes:
            {context}

            Question: {query}

            Answer:"""

            # Use the prompt in the chain
            prompt = PromptTemplate(template=qa_template, input_variables=["context", "query"])

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )

            st.success("RAG pipeline created successfully!")

            return qa_chain

    except Exception as e:
        st.error(f"Error setting up retrieval system: {str(e)}")
        return None