# Overview

PDF Chat Assistant is a Streamlit-based web application that enables interactive conversations with PDF documents using advanced RAG (Retrieval-Augmented Generation) technology. The application supports two RAG approaches:

1. **Traditional Vector RAG**: Uses vector embeddings for semantic search
2. **Knowledge Graph RAG (LightRAG)**: Advanced knowledge graph-based approach using entity-relationship extraction

Users provide their own API keys and base URLs for any OpenAI-compatible LLM service. The system supports 13 preset models plus a custom model input field, allowing users to use any model from their chosen provider.

# User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes (February 2026)
- **User-provided API credentials**: Removed environment variable dependency; users now enter API key and base URL directly in the sidebar
- **Custom model support**: Added text input for custom model IDs alongside the 13 preset models
- **Knowledge Graph visualization**: New tab showing interactive entity-relationship graph (pyvis + networkx) when using LightRAG mode
- **Separate embeddings API config**: Embeddings API key and base URL configurable independently in a collapsible section

## Previous Changes
- Integrated LightRAG knowledge graph-based RAG system as alternative to traditional vector RAG
- Added RAG method selection in sidebar (Traditional Vector RAG vs Knowledge Graph RAG)
- Implemented automatic fallback to traditional RAG if LightRAG fails
- Fixed LightRAG timeout issues (httpx monkey-patching for 600s timeouts)
- Fixed storage folder being cleared on every document upload

# System Architecture

## Frontend Architecture
- **Streamlit Framework**: Web-based interface with sidebar for API config, PDF upload, model selection, and RAG method
- **Three tabs**: Chat, PDF Viewer, Knowledge Graph (graph tab only visible in LightRAG mode)
- **Session State Management**: Stores API credentials, conversation history, processed documents, and processor instances

## Backend Architecture
- **PDF Processing Pipeline**: Uses PyPDF2 and PyMuPDF (fitz) for text extraction, with automatic text chunking
- **Dual RAG System**:
  - **Traditional Vector RAG**: In-memory vector storage using scikit-learn cosine similarity
  - **LightRAG Knowledge Graph**: Entity-relationship extraction using LLM models
- **OpenRouterClient**: Generic OpenAI-compatible API client accepting any base_url
- **Graph Visualization**: graph_visualizer.py reads graphml from LightRAG storage and renders interactive graph via pyvis

## Key Files
- `app.py` - Main Streamlit application
- `openrouter_client.py` - Generic OpenAI-compatible LLM client
- `lightrag_processor.py` - LightRAG integration with user-provided credentials
- `lightrag_timeout_fix.py` - httpx timeout patches for LightRAG v1.3.7+
- `graph_visualizer.py` - Interactive knowledge graph visualization (pyvis + networkx)
- `vector_store.py` - Traditional vector RAG storage
- `topic_extractor.py` - Document topic/keyword extraction
- `pdf_processor.py` - PDF text and image extraction
- `utils.py` / `utils2.py` - Utility functions and AiTunnel embeddings client

## Data Storage Solutions
- **In-Memory Storage**: Vector embeddings and document chunks stored in session state
- **LightRAG Storage**: `./lightrag_storage/` directory contains graphml, vector DBs, and caches
- **No Persistent Database**: Stateless document analysis sessions

## Authentication
- **User-provided API keys**: Entered directly in the sidebar (API key, base URL, optional embeddings API)
- **No environment variables required**: All credentials provided at runtime
- **No user authentication**: Single-user tool

# External Dependencies

## AI/ML Services
- Any OpenAI-compatible API (OpenRouter, OpenAI, etc.) for LLM inference
- Embeddings API (AiTunnel or compatible) for vector representations

## Core Libraries
- **Streamlit**: Web application framework
- **PyPDF2 & PyMuPDF**: PDF text extraction
- **scikit-learn**: Vector similarity calculations
- **NumPy**: Numerical operations
- **LightRAG-HKU**: Knowledge graph-based RAG system
- **networkx**: Graph data structure and graphml parsing
- **pyvis**: Interactive graph visualization
- **LangChain**: Framework components for LLM integration
- **nest-asyncio**: Asyncio event loop management for LightRAG
