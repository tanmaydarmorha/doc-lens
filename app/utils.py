"""
Utility functions for the document analysis project.

This module provides utility functions for accessing language models and embeddings
used throughout the application.
"""

from typing import Any, Dict, Optional

from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings


def get_embedding_model(
    model_name: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
    context_size: int = 2048,
    threads: int = 4,
) -> Embeddings:
    """
    Get an embedding model for document vectorization.
    
    Uses Ollama to access the nomic-embed-text model for generating
    high-quality document embeddings.
    
    Args:
        model_name: Name of the embedding model to use
        base_url: URL of the Ollama server
        context_size: Size of the context window
        threads: Number of threads for computation
        
    Returns:
        An initialized embedding model
        
    Example:
        >>> embedding_model = get_embedding_model()
        >>> embeddings = embedding_model.embed_documents(["Your text here"])
    """
    embedding_model = OllamaEmbeddings(
        model=model_name,
        base_url=base_url,
        num_ctx=context_size,
        num_thread=threads,
    )
    
    return embedding_model


def get_chat_model(
    model_name: str = "llama3.2",
    temperature: float = 0.7,
    top_p: float = 0.9,
    verbose: bool = True,
    response_format: Optional[str] = "json",
    max_tokens: int = 512,
    base_url: str = "http://localhost:11434",
) -> BaseChatModel:
    """
    Get a chat model for interactive document querying.
    
    Uses Ollama to access advanced language models for chat-based
    interactions with documents.
    
    Args:
        model_name: Name of the chat model to use
        temperature: Controls randomness (0.0 to 1.0)
        top_p: Controls diversity via nucleus sampling
        verbose: Whether to print out response text
        response_format: Output format (e.g., "json")
        max_tokens: Maximum number of tokens to generate
        base_url: URL of the Ollama server
        
    Returns:
        An initialized chat model
        
    Example:
        >>> chat_model = get_chat_model()
        >>> response = chat_model.invoke("Summarize this document")
    """
    # Set up callback manager for streaming output
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Initialize the Ollama chat model
    chat_model = ChatOllama(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        callback_manager=callback_manager,
        verbose=verbose,
        format=response_format,
        num_predict=max_tokens,
        repeat_penalty=1.1,
        base_url=base_url,
    )
    
    return chat_model


# Example usage if this file is run directly
if __name__ == "__main__":
    # Simple demonstration of the embedding model
    embed_model = get_embedding_model()
    sample_text = "This is a sample document for testing embeddings."
    
    print(f"Testing embedding model with text: '{sample_text}'")
    try:
        embedding = embed_model.embed_query(sample_text)
        print(f"Successfully generated embedding of dimension {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"Error generating embedding: {e}")
    
    # Simple demonstration of the chat model
    print("\nTesting chat model with a simple query...")
    chat_model = get_chat_model()
    try:
        response = chat_model.invoke("What is document analysis?")
        print(f"Chat model response received successfully")
    except Exception as e:
        print(f"Error using chat model: {e}")