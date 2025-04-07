# Document Analysis & Chat

A powerful document analysis platform that enables users to upload documents, analyze them using embeddings, and interact with them through a chat interface.

## 🚀 Features

- **Document Upload**: Support for TXT, PDF, and DOCX files
- **Intelligent Analysis**: Document chunking and embedding using state-of-the-art models
- **Interactive Chat**: Ask questions about your documents and get AI-powered responses
- **Agentic RAG**: Advanced retrieval-augmented generation with LangGraph for high-quality answers
- **Document Relevance Grading**: Smart evaluation of retrieved documents for better responses
- **Query Reformulation**: Automatic query improvement when initial results are insufficient

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance API framework
- **LangChain**: Framework for LLM application development
- **LangGraph**: Graph-based orchestration for agentic workflows
- **Ollama**: Local LLM integration for embeddings and chat

### Frontend
- **Next.js**: React framework for the web application
- **Streamlit**: Rapid prototyping and development UI

### Databases
- **MongoDB**: User data and document metadata storage
- **Redis Stack**: Vector database for efficient similarity search
- **FAISS**: In-memory vector storage for rapid retrieval

### Models
- **nomic-embed-text**: High-quality document embeddings
- **llama3.2**: Advanced chat model for document interaction

## 📋 Project Structure

```
learning/
├── app/
│   ├── domain/         # Domain models and state definitions
│   ├── services/       # Core services (document loader, LLM service)
│   │   ├── document_loader.py  # Document processing and chunking
│   │   └── llm_service.py      # LangGraph agentic RAG implementation
│   ├── utils.py        # Utility functions for models and embeddings
│   └── run_agent.py    # Streamlit application entry point
├── requirements.txt    # Project dependencies
├── .gitignore          # Git ignore file
└── README.md           # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Ollama installed and running locally
- Required models pulled to Ollama:
  - nomic-embed-text
  - llama3.2

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd learning
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Pull required models to Ollama:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

### Running the Application

#### Streamlit Prototype

Run the Streamlit application:
```bash
streamlit run app/run_agent.py
```

## 📝 Usage

1. Upload a document through the sidebar interface
2. Wait for the document to be processed and embedded
3. Ask questions about your document in the chat interface
4. View AI-generated responses based on the document content

## 🧠 How It Works

The application follows an agentic RAG (Retrieval-Augmented Generation) approach:

1. **Document Processing**: Documents are chunked into manageable pieces
2. **Embedding**: Chunks are embedded using the nomic-embed-text model
3. **Vector Storage**: Embeddings are stored in a FAISS vector store
4. **Query Processing**: User questions trigger the LangGraph workflow:
   - Agent decides whether to retrieve documents
   - Documents are retrieved based on semantic similarity
   - Document relevance is graded
   - If relevant, a response is generated
   - If not relevant, the query is reformulated

## 🔮 Future Enhancements

- User authentication system
- Dashboard for previous analysis runs
- Support for more document formats
- Integration with external knowledge bases
- Improved document visualization

## 📄 License

[MIT License](LICENSE)
