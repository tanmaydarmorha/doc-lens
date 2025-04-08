"""
Streamlit application for document analysis and querying.

This module provides a web interface for uploading documents, 
asking questions about them, and receiving AI-generated responses.
"""

import os
import uuid
import tempfile
import sys
import streamlit as st
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from langchain_community.vectorstores import FAISS
from app.services.document_loader import read_text_and_create_chunks
from app.services.llm_service import LLMService
from app.utils import get_embedding_model
from app.services.embedding_service import create_embeddings_and_save_to_vector_store, retrieve_from_vector_store


def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file to a temporary location and return the path.
    
    Args:
        uploaded_file: The uploaded file from Streamlit
        
    Returns:
        Path to the saved file
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            # Write the uploaded file to the temporary file
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None


def process_document(file_path):
    """
    Process a document file and create embeddings.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Boolean indicating if processing was successful
    """
    # Load the document and split it into chunks
    chunks = read_text_and_create_chunks(file_path)
    
    if not chunks:
        st.error("No content could be extracted from the document.")
        return False
    
    # Create embeddings and save to vector store using the tracking UUID
    create_embeddings_and_save_to_vector_store(chunks, st.session_state.tracking_id)
    
    return True


def main():
    """
    Main function to run the Streamlit application.
    """
    # Set page config
    st.set_page_config(
        page_title="DocLens",
        page_icon="📄",
        layout="wide"
    )
    
    # Initialize session state variables if they don't exist
    if "tracking_id" not in st.session_state:
        st.session_state.tracking_id = str(uuid.uuid4())
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Application title
    st.title("📄 DocLens")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Choose a text file", type=["txt", "md"])
        
        # Only process the document if it's newly uploaded and not already processed
        if uploaded_file is not None and not st.session_state.file_processed:
            # Display a loading spinner while processing the document
            with st.spinner("Processing document..."):
                # Save the uploaded file
                file_path = save_uploaded_file(uploaded_file)
                
                if file_path:
                    # Process the document and create a vector store
                    success = process_document(file_path)
                    
                    if success:
                        st.session_state.file_processed = True
                        st.success(f"Document processed successfully!")
                    
                    # Clean up the temporary file
                    os.unlink(file_path)
        
        # Display the tracking ID
        st.divider()
        st.caption(f"Tracking ID: {st.session_state.tracking_id}")
    
    # Main content area
    if st.session_state.file_processed:
        # Display chat interface
        st.header("Ask questions about your document")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
        
        # Input for user question
        user_question = st.chat_input("Ask a question about your document...")
        
        if user_question:
            # Add user question to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.chat_message("user").write(user_question)
            
            # Create a placeholder for the assistant's response with a spinner
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Thinking..."):
                    try:
                        # Retrieve the vector store using the tracking UUID
                        vectorstore = retrieve_from_vector_store(st.session_state.tracking_id)
                        # Initialize the LLM service with the vector store
                        llm_service = LLMService(vectorstore=vectorstore)
                        
                        # Get the response from the LLM service
                        response = llm_service.query(
                            question=user_question,
                            tracking_id=st.session_state.tracking_id
                        )
                        
                        # Display the response
                        message_placeholder.write(response)
                    except FileNotFoundError as e:
                        error_message = "Vector store not found. Please upload a document first."
                        message_placeholder.error(error_message)
                        response = error_message
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    else:
        # Display instructions when no file is uploaded
        st.info("👈 Please upload a document to get started.")
        
        # Example questions
        st.header("Example questions you can ask:")
        examples = [
            "What is the main topic of this document?",
            "Can you summarize the key points?",
            "What are the conclusions or recommendations?",
            "Explain the concept of [topic] mentioned in the document."
        ]
        for example in examples:
            st.markdown(f"- {example}")


if __name__ == "__main__":
    main()