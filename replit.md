# Overview

PDF Chat Assistant is a Streamlit-based web application that enables interactive conversations with PDF documents using RAG (Retrieval-Augmented Generation) technology. The application processes uploaded PDF files, creates vector embeddings for semantic search, and leverages multiple LLM models through OpenRouter API to provide contextual answers based on document content.

The system supports 13 different LLM models including GPT-4, Claude 3.5, Gemini Pro, Llama 3.1, and others, allowing users to choose their preferred AI model for document analysis and conversation.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Framework**: Web-based interface with drag-and-drop PDF upload, real-time chat interface, and model selection sidebar
- **Session State Management**: Maintains conversation history, processed documents, and vector store instances across user interactions
- **Component Structure**: Modular design with separate components for PDF processing, vector operations, and API interactions

## Backend Architecture
- **PDF Processing Pipeline**: Uses PyPDF2 and PyMuPDF (fitz) for text extraction, with automatic text chunking and overlap management for optimal retrieval
- **Vector Store System**: In-memory vector storage using scikit-learn for cosine similarity calculations, with AiTunnel API for embeddings generation
- **RAG Implementation**: Retrieves relevant document chunks based on user queries and constructs contextual prompts for LLM models
- **Topic Extraction**: Automated keyword and topic identification using frequency analysis and LLM-based content analysis

## Data Storage Solutions
- **In-Memory Storage**: Vector embeddings and document chunks stored in session state for single-session persistence
- **Temporary File Handling**: PDF files processed through temporary storage with automatic cleanup
- **No Persistent Database**: Current architecture operates without permanent data storage, suitable for stateless document analysis sessions

## Authentication and Authorization
- **API Key Management**: Environment variable-based configuration for OpenRouter and AiTunnel API keys
- **No User Authentication**: Application operates as a single-user tool without login requirements
- **API Rate Limiting**: Handled at the external service level (OpenRouter/AiTunnel)

# External Dependencies

## AI/ML Services
- **OpenRouter API**: Primary LLM service providing access to 13 different models (GPT-4o, Claude 3.5 Sonnet, Gemini Pro 1.5, Llama 3.1, etc.)
- **AiTunnel API**: Embeddings generation service for vector representations of document chunks
- **Model Support**: OpenAI, Anthropic, Google, Meta, Mistral, and Qwen model families

## Core Libraries
- **Streamlit**: Web application framework for the user interface
- **PyPDF2 & PyMuPDF**: PDF text extraction and processing
- **scikit-learn**: Vector similarity calculations and machine learning utilities
- **NumPy**: Numerical operations for vector computations
- **LangChain**: Framework components for LLM integration and document processing

## Development Tools
- **UV Package Manager**: Modern Python dependency management
- **Environment Configuration**: dotenv for API key management
- **Type Hints**: Full typing support for better code maintainability

## Optional Integrations
- **Image Processing**: PIL for potential image extraction from PDFs
- **ChromaDB/FAISS**: Alternative vector databases (installed but not actively used in current implementation)
- **Export/Import**: JSON-based chat history persistence capabilities