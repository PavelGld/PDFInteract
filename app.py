"""
PDF Chat Assistant - Главное приложение

Streamlit веб-приложение для интерактивного чата с PDF документами.
Использует RAG (Retrieval-Augmented Generation) для поиска релевантного контекста
и генерации ответов через OpenRouter API.

Основные компоненты:
- PDF загрузка и обработка
- Векторное хранилище с эмбеддингами
- LLM интеграция через OpenRouter
- Пользовательский интерфейс чата
"""

import streamlit as st
import os
import tempfile
from io import BytesIO
import base64

from pdf_processor import PDFProcessor
from vector_store import VectorStore
from openrouter_client import OpenRouterClient
from utils import validate_pdf_file, format_chat_message

# Page configuration
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Initialize components
@st.cache_resource
def get_pdf_processor():
    return PDFProcessor()

@st.cache_resource
def get_openrouter_client():
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        st.error("⚠️ OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable.")
        st.stop()
    return OpenRouterClient(api_key)

pdf_processor = get_pdf_processor()
openrouter_client = get_openrouter_client()

# Sidebar for PDF upload and settings
with st.sidebar:
    st.header("📄 PDF Upload")
    
    # File upload first
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF file (max 20MB) to start chatting with it",
        accept_multiple_files=False
    )
    
    st.divider()
    
    # Model selection
    st.header("🤖 LLM Model")
    model_options = {
        "GPT-3.5 Turbo": "openai/gpt-3.5-turbo",
        "Claude 3 Sonnet": "anthropic/claude-3-sonnet-20240229",
        "Claude 3 Opus": "anthropic/claude-3-opus-20240229",
        "GPT-4": "openai/gpt-4",
        "GPT-4 Turbo": "openai/gpt-4-turbo"
    }
    
    selected_model = st.selectbox(
        "Select model:",
        options=list(model_options.keys()),
        index=0,
        help="Choose the language model for processing your questions"
    )
    
    st.session_state.selected_model = model_options[selected_model]
    
    st.divider()
    
    # Settings section
    st.header("⚙️ Settings")
    
    # Debug mode toggle
    debug_mode = st.checkbox(
        "Show debug information", 
        value=st.session_state.get('debug_mode', False),
        help="Display technical details about document processing and search results"
    )
    st.session_state.debug_mode = debug_mode
    
    st.divider()
    
    # Chat export/import
    st.header("💾 Chat History")
    
    # Download chat history
    if st.session_state.messages and len(st.session_state.messages) > 0:
        chat_history = []
        for msg in st.session_state.messages:
            chat_history.append(f"{msg['role'].title()}: {msg['content']}")
        
        chat_text = "\n\n".join(chat_history)
        st.download_button(
            label="📥 Download Chat History",
            data=chat_text,
            file_name=f"chat_history_{st.session_state.pdf_name or 'session'}.txt",
            mime="text/plain",
            help="Download the conversation as a text file"
        )
    
    # Upload chat history - fix infinite loading issue
    uploaded_chat = st.file_uploader(
        "📤 Upload Chat History",
        type="txt",
        help="Upload a previously saved chat history",
        key="chat_upload"
    )
    
    if uploaded_chat is not None and not st.session_state.get('chat_imported', False):
        try:
            chat_content = uploaded_chat.read().decode('utf-8')
            # Parse chat history
            lines = chat_content.split('\n\n')
            imported_messages = []
            
            for line in lines:
                if line.strip():
                    if line.startswith('User: '):
                        imported_messages.append({"role": "user", "content": line[6:]})
                    elif line.startswith('Assistant: '):
                        imported_messages.append({"role": "assistant", "content": line[11:]})
            
            if imported_messages:
                st.session_state.messages = imported_messages
                st.session_state.chat_imported = True
                st.success(f"✅ Imported {len(imported_messages)} messages")
                st.rerun()
        except Exception as e:
            st.error(f"❌ Error importing chat: {str(e)}")
    
    # Reset chat import flag when no file is uploaded
    if uploaded_chat is None:
        st.session_state.chat_imported = False
    
    if uploaded_file is not None:
        # Validate file
        is_valid, error_message = validate_pdf_file(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ {error_message}")
        else:
            # Check if this is a new file
            if st.session_state.pdf_name != uploaded_file.name:
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pdf_processed = False
                st.session_state.messages = []
                st.session_state.vector_store = None
                
                with st.spinner("🔄 Processing PDF..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Extract text from PDF
                        text_content = pdf_processor.extract_text(tmp_file_path)
                        
                        if not text_content.strip():
                            st.error("❌ No text content found in the PDF. Please ensure the PDF contains readable text.")
                        else:
                            # Create chunks
                            chunks = pdf_processor.create_chunks(text_content)
                            
                            # Create vector store with API embeddings
                            api_key = os.getenv("OPENROUTER_API_KEY", "")
                            vector_store = VectorStore(api_key=api_key)
                            vector_store.add_chunks(chunks)
                            
                            # Store PDF as base64 for viewer
                            pdf_base64 = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
                            
                            st.session_state.vector_store = vector_store
                            st.session_state.pdf_content = text_content
                            st.session_state.pdf_processed = True
                            st.session_state.pdf_base64 = pdf_base64
                            
                            st.success(f"✅ PDF processed successfully! Found {len(chunks)} text chunks.")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing PDF: {str(e)}")
                    finally:
                        # Clean up temporary file
                        if 'tmp_file_path' in locals():
                            os.unlink(tmp_file_path)
    
    # Main content area with tabs instead of columns
    if st.session_state.pdf_processed and st.session_state.pdf_name:
        st.success(f"📚 **Current PDF:** {st.session_state.pdf_name}")
    
    # Clear conversation button
    if st.session_state.messages:
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.messages = []
            st.rerun()

# Main content area
st.title("📚 PDF Chat Assistant")
st.markdown("Upload a PDF document and start an interactive conversation about its content using AI.")

# Display current status
if not st.session_state.pdf_processed:
    st.info("👈 Please upload a PDF file from the sidebar to start chatting.")
else:
    # Create tabs for main interface
    chat_tab, pdf_tab = st.tabs(["💬 Chat with PDF", "📄 View PDF"])
    
    with pdf_tab:
        st.markdown("### PDF Document")
        if st.session_state.get('pdf_base64'):
            # Show PDF stats only in debug mode
            if st.session_state.get('debug_mode', False) and st.session_state.pdf_content:
                word_count = len(st.session_state.pdf_content.split())
                char_count = len(st.session_state.pdf_content)
                st.info(f"📊 **Stats:** {word_count:,} words, {char_count:,} characters")
            
            pdf_display = f"""
            <iframe src="data:application/pdf;base64,{st.session_state.pdf_base64}" 
                    width="100%" height="800px" style="border: 1px solid #ddd; border-radius: 8px;">
                <p>Your browser doesn't support PDF viewing. 
                <a href="data:application/pdf;base64,{st.session_state.pdf_base64}">Download the PDF</a> to view it.</p>
            </iframe>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.info("PDF content will appear here after upload")
    
    with chat_tab:
        st.markdown("### Chat Interface")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDF..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Search for relevant chunks using LangChain FAISS
                        relevant_chunks = st.session_state.vector_store.search(prompt, k=5, score_threshold=0.5)
                        
                        # Debug information - show only if debug mode is enabled
                        if st.session_state.get('debug_mode', False):
                            stats = st.session_state.vector_store.get_stats()
                            total_chunks = stats.get('total_chunks', 0)
                            total_docs = stats.get('total_documents', 0)
                            vectorstore_active = stats.get('vectorstore_active', False)
                            
                            st.write(f"**Debug:** Всего фрагментов: {total_chunks}, документов: {total_docs}")
                            st.write(f"**Debug:** Векторное хранилище: {'Активно' if vectorstore_active else 'Неактивно'}")
                            st.write(f"**Debug:** Найдено {len(relevant_chunks)} релевантных фрагментов")
                            
                            if relevant_chunks:
                                st.write("**Найденные релевантные фрагменты:**")
                                for i, chunk in enumerate(relevant_chunks[:2]):
                                    st.write(f"**Фрагмент {i+1}** (Similarity: {chunk.get('score', 0):.3f}, Distance: {chunk.get('distance', 0):.3f}):")
                                    st.write(f"```\n{chunk['content'][:400]}...\n```")
                        
                        # Prepare context from relevant chunks
                        if relevant_chunks:
                            context = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
                            if st.session_state.get('debug_mode', False):
                                st.write(f"**Debug:** Используется {len(relevant_chunks)} релевантных фрагментов")
                        else:
                            # If no relevant chunks found, use lower threshold
                            if st.session_state.get('debug_mode', False):
                                st.write("**Debug:** Снижаем порог поиска...")
                            fallback_chunks = st.session_state.vector_store.search(prompt, k=3, score_threshold=0.1)
                            if fallback_chunks:
                                context = "\n\n".join([chunk["content"] for chunk in fallback_chunks])
                                if st.session_state.get('debug_mode', False):
                                    for i, chunk in enumerate(fallback_chunks[:2]):
                                        st.write(f"**Запасной фрагмент {i+1}** (Score: {chunk.get('score', 0):.3f}):")
                                        st.write(f"```\n{chunk['content'][:300]}...\n```")
                            else:
                                # Final fallback
                                context = "\n\n".join([chunk["content"] for chunk in st.session_state.vector_store.chunks[:3]])
                                if st.session_state.get('debug_mode', False):
                                    st.write("**Debug:** Используем первые фрагменты документа")
                        
                        # Show context length only in debug mode
                        if st.session_state.get('debug_mode', False):
                            st.write(f"**Debug:** Размер контекста: {len(context)} символов")
                        
                        # Get response from OpenRouter
                        response = openrouter_client.get_response(
                            messages=st.session_state.messages[:-1],  # Exclude the current question
                            question=prompt,
                            context=context,
                            model=st.session_state.selected_model
                        )
                        
                        # Display response
                        st.markdown(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Show source information
                        if relevant_chunks:
                            with st.expander("📖 Source Information", expanded=False):
                                st.write("**Relevant sections from your PDF:**")
                                for i, chunk in enumerate(relevant_chunks[:3], 1):
                                    st.write(f"**Section {i}** (Relevance: {chunk.get('score', 0):.2f})")
                                    st.write(chunk['content'][:500] + ("..." if len(chunk['content']) > 500 else ""))
                                    st.divider()
                    
                    except Exception as e:
                        error_msg = f"❌ Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Built with Streamlit • Powered by OpenRouter API • Vector search with FAISS
    </div>
    """, 
    unsafe_allow_html=True
)
