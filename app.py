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
from topic_extractor import TopicExtractor
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
    st.session_state.pdf_content = ""
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "pdf_base64" not in st.session_state:
    st.session_state.pdf_base64 = ""
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "openai/gpt-4o"
if "document_topics" not in st.session_state:
    st.session_state.document_topics = []
if "document_images" not in st.session_state:
    st.session_state.document_images = []
if "document_summary" not in st.session_state:
    st.session_state.document_summary = ""

# Helper functions
def get_pdf_processor():
    """Get PDF processor instance."""
    return PDFProcessor()

@st.cache_resource
def get_openrouter_client():
    """Get cached OpenRouter client instance."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        st.error("⚠️ OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable.")
        st.stop()
    return OpenRouterClient(api_key)

pdf_processor = get_pdf_processor()
openrouter_client = get_openrouter_client()
topic_extractor = TopicExtractor(openrouter_client)

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
        "GPT-4o 🖼️ (Vision)": "openai/gpt-4o",
        "GPT-4o Mini 🖼️ (Vision)": "openai/gpt-4o-mini", 
        "GPT-4 Turbo 🖼️ (Vision)": "openai/gpt-4-turbo",
        "GPT-3.5 Turbo": "openai/gpt-3.5-turbo",
        "Claude 3.5 Sonnet 🖼️ (Vision)": "anthropic/claude-3-5-sonnet-20241022",
        "Claude 3 Opus 🖼️ (Vision)": "anthropic/claude-3-opus-20240229",
        "Claude 3 Haiku 🖼️ (Vision)": "anthropic/claude-3-haiku-20240307",
        "Gemini Pro 1.5 🖼️ (Vision)": "google/gemini-pro-1.5",
        "Gemini Flash 1.5 🖼️ (Vision)": "google/gemini-flash-1.5",
        "Llama 3.2 90B Vision 🖼️": "meta-llama/llama-3.2-90b-vision-instruct",
        "Llama 3.1 405B": "meta-llama/llama-3.1-405b-instruct",
        "Llama 3.1 70B": "meta-llama/llama-3.1-70b-instruct",
        "Mixtral 8x7B": "mistralai/mixtral-8x7b-instruct",
        "Qwen 2.5 72B": "qwen/qwen-2.5-72b-instruct"
    }
    
    selected_model = st.selectbox(
        "Select model:",
        options=list(model_options.keys()),
        index=0,
        help="🖼️ Vision models can analyze images from PDFs"
    )
    
    # Show vision capability info
    if "🖼️" in selected_model:
        st.info("🖼️ Эта модель может анализировать изображения из PDF")
    else:
        st.warning("⚠️ Эта модель работает только с текстом")
    
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
    
    # Chat history management - show when PDF is processed
    if st.session_state.pdf_processed:
        st.divider()
        st.header("💾 Chat History")
        
        # Export chat history (always available)
        if st.session_state.messages:
            chat_text = ""
            for msg in st.session_state.messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_text += f"{role}: {msg['content']}\n\n"
            
            st.download_button(
                label="📥 Download Chat History",
                data=chat_text,
                file_name=f"chat_history_{st.session_state.pdf_name.replace('.pdf', '')}.txt",
                mime="text/plain",
                help="Download chat history as text file"
            )
        else:
            st.info("💬 Start chatting to enable download")
        
        # Import chat history (always available)
        st.markdown("**Import Chat History:**")
        uploaded_chat = st.file_uploader(
            "Upload previous chat history",
            type="txt",
            help="Upload a previously exported chat history file",
            key="chat_upload"
        )
        
        if uploaded_chat is not None:
            try:
                # Read the uploaded file
                chat_content = uploaded_chat.read().decode("utf-8")
                
                # Parse the chat content
                new_messages = []
                lines = chat_content.strip().split('\n\n')
                
                for line in lines:
                    if line.strip():
                        if line.startswith("User: "):
                            content = line[6:]  # Remove "User: "
                            new_messages.append({"role": "user", "content": content})
                        elif line.startswith("Assistant: "):
                            content = line[11:]  # Remove "Assistant: "
                            new_messages.append({"role": "assistant", "content": content})
                
                if new_messages:
                    if st.button("📤 Import Chat", help="Replace current chat with imported history"):
                        st.session_state.messages = new_messages
                        st.success(f"✅ Imported {len(new_messages)} messages successfully!")
                        st.rerun()
                    
                    st.info(f"📋 Preview: Found {len(new_messages)} messages in the file")
                
            except Exception as e:
                st.error(f"❌ Error reading chat file: {str(e)}")
        
        # Clear chat button
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat", help="Clear all chat messages"):
                st.session_state.messages = []
                st.success("✅ Chat cleared!")
                st.rerun()
    
    # Process uploaded file
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("🔄 Processing PDF..."):
                try:
                    # Validate file
                    is_valid, error_msg = validate_pdf_file(uploaded_file)
                    if not is_valid:
                        st.error(f"❌ {error_msg}")
                    else:
                        # Create temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        # Extract text
                        text_content = pdf_processor.extract_text(tmp_file_path)
                        
                        if not text_content.strip():
                            st.error("❌ Could not extract text from PDF. The file might be image-based or corrupted.")
                        else:
                            # Create chunks
                            chunks = pdf_processor.create_chunks(text_content)
                            
                            # Extract images from PDF
                            images = pdf_processor.extract_images(tmp_file_path)
                            
                            # Analyze images (basic description for now)
                            images_description = ""
                            if images:
                                images_description = f"Документ содержит {len(images)} изображений"
                            
                            # Extract topics using LLM
                            topics = topic_extractor.extract_topics_llm(text_content, images_description, st.session_state.selected_model)
                            
                            # Generate document summary
                            summary = topic_extractor.generate_document_summary(text_content, topics)
                            
                            # Create vector store with Course API
                            course_api_key = os.environ.get("COURSE_API_KEY")
                            if not course_api_key:
                                st.error("⚠️ Course API key not found. Please set COURSE_API_KEY environment variable.")
                                st.stop()
                            
                            vector_store = VectorStore(course_api_key)
                            vector_store.add_chunks(chunks)
                            
                            # Store PDF as base64 for viewing
                            pdf_base64 = base64.b64encode(uploaded_file.getvalue()).decode()
                            
                            # Update session state
                            st.session_state.pdf_processed = True
                            st.session_state.vector_store = vector_store
                            st.session_state.pdf_content = text_content
                            st.session_state.pdf_name = uploaded_file.name
                            st.session_state.pdf_base64 = pdf_base64
                            st.session_state.document_topics = topics
                            st.session_state.document_images = images
                            st.session_state.document_summary = summary
                            st.session_state.messages = []  # Clear previous chat
                            
                            success_msg = f"✅ Successfully processed {len(chunks)} text chunks"
                            if images:
                                success_msg += f" and {len(images)} images"
                            if topics:
                                success_msg += f". Topics: {', '.join(topics[:3])}{'...' if len(topics) > 3 else ''}"
                            
                            st.success(success_msg)
                            st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                finally:
                    # Clean up temporary file
                    if 'tmp_file_path' in locals():
                        os.unlink(tmp_file_path)

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
        
        # Show document topics and summary
        if st.session_state.get('document_topics'):
            st.markdown("**🏷️ Тематические метки:**")
            # Display topics as colored badges
            topics_html = ""
            for topic in st.session_state.document_topics:
                topics_html += f'<span style="background-color: #e1f5fe; color: #01579b; padding: 2px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 0.85em;">{topic}</span>'
            st.markdown(topics_html, unsafe_allow_html=True)
            st.markdown("")
        
        # Show document summary
        if st.session_state.get('document_summary'):
            st.markdown("**📋 Краткое описание:**")
            st.info(st.session_state.document_summary)
        
        # Show images info
        if st.session_state.get('document_images'):
            with st.expander(f"🖼️ Изображения в документе ({len(st.session_state.document_images)})", expanded=False):
                for i, img in enumerate(st.session_state.document_images[:5]):  # Show first 5
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        try:
                            # Display small thumbnail
                            img_data = base64.b64decode(img['data'])
                            st.image(img_data, width=100, caption=f"Страница {img['page']}")
                        except:
                            st.write(f"📄 Страница {img['page']}")
                    with col2:
                        st.write(f"**Размер:** {img['width']}×{img['height']} пикселей")
                        st.write(f"**Размер файла:** {img['size_bytes']:,} байт")
        
        if st.session_state.get('pdf_base64'):
            # Show PDF stats only in debug mode
            if st.session_state.get('debug_mode', False) and st.session_state.pdf_content:
                word_count = len(st.session_state.pdf_content.split())
                char_count = len(st.session_state.pdf_content)
                st.info(f"📊 **Stats:** {word_count:,} words, {char_count:,} characters")
            
            st.markdown("---")
            pdf_display = f"""
            <iframe src="data:application/pdf;base64,{st.session_state.pdf_base64}" 
                    width="100%" height="600px" style="border: 1px solid #ddd; border-radius: 8px;">
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
                        # Search for relevant chunks
                        relevant_chunks = st.session_state.vector_store.search(prompt, k=5, score_threshold=0.5)
                        
                        # Debug information - show only if debug mode is enabled
                        if st.session_state.get('debug_mode', False):
                            stats = st.session_state.vector_store.get_stats()
                            total_chunks = stats.get('total_chunks', 0)
                            st.info(f"🔍 **Search Results:** Found {len(relevant_chunks)} relevant chunks from {total_chunks} total chunks")
                            
                            if relevant_chunks:
                                with st.expander("📄 View Retrieved Context", expanded=False):
                                    for i, chunk in enumerate(relevant_chunks[:3]):  # Show top 3
                                        st.write(f"**Chunk {i+1}** (Score: {chunk.get('score', 0):.3f})")
                                        st.write(chunk.get('content', '')[:300] + "...")
                                        st.divider()
                        
                        if relevant_chunks:
                            # Combine contexts
                            context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
                            
                            # Get response from OpenRouter
                            response = openrouter_client.get_response(
                                messages=st.session_state.messages[:-1],  # Exclude current message
                                question=prompt,
                                context=context,
                                model=st.session_state.selected_model,
                                max_tokens=1000,
                                temperature=0.7
                            )
                            
                            st.markdown(response)
                            
                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            # No relevant context found
                            response = "I couldn't find relevant information in the document to answer your question. Could you try rephrasing your question or ask about something more specific from the PDF content?"
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
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