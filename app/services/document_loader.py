import os
from typing import List, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def read_text_and_create_chunks(text_file: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Read a text file and split its content into chunks.
    
    Args:
        text_file: Path to the text file
        chunk_size: The size of each text chunk
        chunk_overlap: The overlap between consecutive chunks
        
    Returns:
        A list of Document objects containing the chunked text
    """
    try:
        # Read the text file
        with open(text_file, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Print the extracted text for now
        print("Extracted text content:")
        print(text[:500] + "..." if len(text) > 500 else text)  # Print first 500 chars for preview
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Split text into chunks
        chunks = text_splitter.create_documents([text])
        
        return chunks
        
    except Exception as e:
        print(f"Error processing text file: {str(e)}")
        return []

def main():
    """
    Main function to demonstrate text file reading and chunking functionality.
    Uses message.txt as the input file.
    """
    # Get the absolute path to the text file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    text_path = os.path.join(current_dir, "resources", "message.txt")
    
    # Check if the file exists
    if not os.path.exists(text_path):
        print(f"Error: File not found at {text_path}")
        return
    
    print(f"Reading text file from: {text_path}")
    
    # Process the text file
    chunks = read_text_and_create_chunks(text_path)
    
    # Print information about the chunks
    print(f"\nCreated {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {len(chunk.page_content)} characters")
        # Print a preview of each chunk (first 100 characters)
        preview = chunk.page_content[:100].replace('\n', ' ') + "..." if len(chunk.page_content) > 100 else chunk.page_content
        print(f"Preview: {preview}")
        print("-" * 50)

if __name__ == "__main__":
    main()