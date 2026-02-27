"""
PDF Chat Assistant - Main Application

Streamlit web application for interactive chat with PDF documents.
Uses RAG (Retrieval-Augmented Generation) with user-provided API credentials.
Supports Traditional Vector RAG and Knowledge Graph RAG (LightRAG).
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

try:
    from lightrag_processor import create_lightrag_processor, run_async_insert, run_async_query, run_async_initialize
    LIGHTRAG_AVAILABLE = True
except ImportError as e:
    print(f"LightRAG not available: {e}")
    LIGHTRAG_AVAILABLE = False

try:
    from graph_visualizer import render_knowledge_graph
    GRAPH_VIZ_AVAILABLE = True
except ImportError as e:
    print(f"Graph visualization not available: {e}")
    GRAPH_VIZ_AVAILABLE = False

st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
if "lightrag_processor" not in st.session_state:
    st.session_state.lightrag_processor = None
if "rag_method" not in st.session_state:
    st.session_state.rag_method = "Knowledge Graph RAG (LightRAG)"
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
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "base_url" not in st.session_state:
    st.session_state.base_url = "https://openrouter.ai/api/v1"
if "embeddings_api_key" not in st.session_state:
    st.session_state.embeddings_api_key = ""
if "embeddings_base_url" not in st.session_state:
    st.session_state.embeddings_base_url = "https://api.aitunnel.ru/v1/"

pdf_processor = PDFProcessor()

with st.sidebar:
    st.header("🔑 API Settings")

    api_key = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        help="API key for LLM service (e.g. OpenRouter, OpenAI-compatible)",
        placeholder="sk-..."
    )
    st.session_state.api_key = api_key

    base_url = st.text_input(
        "Base URL",
        value=st.session_state.base_url,
        help="Base URL for LLM API endpoint",
        placeholder="https://openrouter.ai/api/v1"
    )
    st.session_state.base_url = base_url

    with st.expander("Embeddings API (for Vector RAG & LightRAG)", expanded=False):
        embeddings_api_key = st.text_input(
            "Embeddings API Key",
            value=st.session_state.embeddings_api_key,
            type="password",
            help="API key for embeddings service. If empty, the main API key is used.",
            placeholder="sk-aitunnel-..."
        )
        st.session_state.embeddings_api_key = embeddings_api_key

        embeddings_base_url = st.text_input(
            "Embeddings Base URL",
            value=st.session_state.embeddings_base_url,
            help="Base URL for embeddings API",
            placeholder="https://api.aitunnel.ru/v1/"
        )
        st.session_state.embeddings_base_url = embeddings_base_url

    api_configured = bool(api_key)

    st.divider()

    st.header("📄 PDF Upload")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF file (max 50MB) to start chatting with it",
        accept_multiple_files=False
    )

    st.divider()

    st.header("🧠 RAG Method")
    rag_method = st.radio(
        "Choose RAG approach:",
        ["Traditional Vector RAG", "Knowledge Graph RAG (LightRAG)"],
        index=1,
        help="Traditional RAG uses vector similarity search. Knowledge Graph RAG builds entity-relationship graphs for better context understanding."
    )

    st.session_state.rag_method = rag_method

    st.divider()

    st.header("🤖 LLM Model")
    model_options = {
        "GPT-4o": "openai/gpt-4o",
        "GPT-4o Mini": "openai/gpt-4o-mini",
        "GPT-4 Turbo": "openai/gpt-4-turbo",
        "GPT-3.5 Turbo": "openai/gpt-3.5-turbo",
        "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet-20241022",
        "Claude 3 Opus": "anthropic/claude-3-opus-20240229",
        "Claude 3 Haiku": "anthropic/claude-3-haiku-20240307",
        "Gemini Pro 1.5": "google/gemini-pro-1.5",
        "Gemini Flash 1.5": "google/gemini-flash-1.5",
        "Llama 3.1 405B": "meta-llama/llama-3.1-405b-instruct",
        "Llama 3.1 70B": "meta-llama/llama-3.1-70b-instruct",
        "Mixtral 8x7B": "mistralai/mixtral-8x7b-instruct",
        "Qwen 2.5 72B": "qwen/qwen-2.5-72b-instruct"
    }

    selected_model = st.selectbox(
        "Select model:",
        options=list(model_options.keys()),
        index=0,
        help="Choose your preferred language model for chat"
    )

    custom_model = st.text_input(
        "Or enter custom model ID:",
        value="",
        help="Enter a custom model identifier (e.g. 'mistralai/mistral-large-latest'). If filled, this overrides the selection above.",
        placeholder="provider/model-name"
    )

    if custom_model.strip():
        st.session_state.selected_model = custom_model.strip()
    else:
        st.session_state.selected_model = model_options[selected_model]

    st.divider()

    st.header("⚙️ Settings")

    debug_mode = st.checkbox(
        "Show debug information",
        value=st.session_state.get('debug_mode', False),
        help="Display technical details about document processing and search results"
    )
    st.session_state.debug_mode = debug_mode

    if st.session_state.pdf_processed:
        st.divider()
        st.header("💾 Chat History")

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

        st.markdown("**Import Chat History:**")
        uploaded_chat = st.file_uploader(
            "Upload previous chat history",
            type="txt",
            help="Upload a previously exported chat history file",
            key="chat_upload"
        )

        if uploaded_chat is not None:
            try:
                chat_content = uploaded_chat.read().decode("utf-8")
                new_messages = []
                lines = chat_content.strip().split('\n\n')

                for line in lines:
                    if line.strip():
                        if line.startswith("User: "):
                            content = line[6:]
                            new_messages.append({"role": "user", "content": content})
                        elif line.startswith("Assistant: "):
                            content = line[11:]
                            new_messages.append({"role": "assistant", "content": content})

                if new_messages:
                    if st.button("📤 Import Chat", help="Replace current chat with imported history"):
                        st.session_state.messages = new_messages
                        st.success(f"✅ Imported {len(new_messages)} messages successfully!")
                        st.rerun()

                    st.info(f"📋 Preview: Found {len(new_messages)} messages in the file")

            except Exception as e:
                st.error(f"❌ Error reading chat file: {str(e)}")

        if st.session_state.messages:
            if st.button("🗑️ Clear Chat", help="Clear all chat messages"):
                st.session_state.messages = []
                st.success("✅ Chat cleared!")
                st.rerun()

    if uploaded_file is not None and api_configured:
        if uploaded_file.name != st.session_state.pdf_name:
            if st.session_state.lightrag_processor is not None:
                st.session_state.lightrag_processor.clear_storage()
                st.session_state.lightrag_processor = None
            with st.spinner("🔄 Processing PDF..."):
                try:
                    is_valid, error_msg = validate_pdf_file(uploaded_file)
                    if not is_valid:
                        st.error(f"❌ {error_msg}")
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        text_content = pdf_processor.extract_text(tmp_file_path)

                        if not text_content.strip():
                            st.error("❌ Could not extract text from PDF. The file might be image-based or corrupted.")
                        else:
                            chunks = pdf_processor.create_chunks(text_content)
                            images = pdf_processor.extract_images(tmp_file_path)
                            images_description = ""
                            if images:
                                images_description = f"Документ содержит {len(images)} изображений"

                            openrouter_client = OpenRouterClient(api_key, base_url)
                            topic_extractor = TopicExtractor(openrouter_client)

                            topics = topic_extractor.extract_topics_llm(text_content, images_description, st.session_state.selected_model)
                            summary = topic_extractor.generate_document_summary(text_content, topics)

                            if st.session_state.rag_method == "Knowledge Graph RAG (LightRAG)" and LIGHTRAG_AVAILABLE:
                                lightrag_proc = create_lightrag_processor(
                                    api_key=api_key,
                                    base_url=base_url,
                                    embeddings_api_key=st.session_state.embeddings_api_key or api_key,
                                    embeddings_base_url=st.session_state.embeddings_base_url,
                                    model=st.session_state.selected_model
                                )

                                st.info("🔗 Testing API connection...")
                                conn_ok, conn_err = lightrag_proc.test_connection()
                                if not conn_ok:
                                    st.error(f"❌ API connection failed: {conn_err}")
                                    st.error("Please check your API Key, Base URL, and Model settings in the sidebar.")
                                    st.stop()

                                st.info("🧠 Building knowledge graph with LightRAG...")

                                with st.spinner("Initializing knowledge graph..."):
                                    try:
                                        run_async_initialize(lightrag_proc)
                                        success = run_async_insert(lightrag_proc, text_content, uploaded_file.name)

                                        if success:
                                            st.session_state.lightrag_processor = lightrag_proc
                                            st.session_state.vector_store = None
                                            st.success("✅ Knowledge graph built successfully!")
                                        else:
                                            st.error("❌ Failed to build knowledge graph. Falling back to traditional RAG.")
                                            st.session_state.rag_method = "Traditional Vector RAG"
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"❌ LightRAG error: {str(e)}. Falling back to traditional RAG.")
                                        st.session_state.rag_method = "Traditional Vector RAG"
                                        st.rerun()

                            if st.session_state.rag_method == "Traditional Vector RAG" or not LIGHTRAG_AVAILABLE:
                                if not LIGHTRAG_AVAILABLE:
                                    st.warning("⚠️ LightRAG not available. Using traditional vector RAG.")

                                emb_key = st.session_state.embeddings_api_key
                                if not emb_key:
                                    st.error("⚠️ Embeddings API key is required for Traditional Vector RAG. Please fill it in the sidebar.")
                                    st.stop()

                                st.info(f"🔄 Creating vector embeddings for {len(chunks)} text chunks...")
                                st.info("⏳ This process will take about 1 second per chunk (optimized for API limits)")

                                progress_bar = st.progress(0)
                                status_text = st.empty()

                                vector_store = VectorStore(emb_key)

                                def update_progress(current, total):
                                    progress = current / total if total > 0 else 0
                                    progress_bar.progress(progress)
                                    remaining = total - current
                                    est_time_remaining = remaining * 1
                                    status_text.text(f"Processing {current}/{total} chunks... (~{est_time_remaining//60}m {est_time_remaining%60}s remaining)")

                                vector_store.add_chunks(chunks, progress_callback=update_progress)

                                progress_bar.empty()
                                status_text.empty()

                                st.session_state.vector_store = vector_store
                                st.session_state.lightrag_processor = None

                            pdf_base64 = base64.b64encode(uploaded_file.getvalue()).decode()

                            st.session_state.pdf_processed = True
                            st.session_state.pdf_content = text_content
                            st.session_state.pdf_name = uploaded_file.name
                            st.session_state.pdf_base64 = pdf_base64
                            st.session_state.document_topics = topics
                            st.session_state.document_images = images
                            st.session_state.document_summary = summary
                            st.session_state.messages = []

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
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
    elif uploaded_file is not None and not api_configured:
        st.warning("⚠️ Please enter your API key above before uploading a PDF.")

if st.session_state.messages:
    if st.button("🗑️ Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()

st.title("📚 PDF Chat Assistant")
st.markdown("Upload a PDF document and start an interactive conversation about its content using AI.")

if not api_configured:
    st.warning("👈 Please enter your API key in the sidebar to get started.")
elif not st.session_state.pdf_processed:
    st.info("👈 Please upload a PDF file from the sidebar to start chatting.")
else:
    use_lightrag = st.session_state.lightrag_processor is not None
    if use_lightrag and GRAPH_VIZ_AVAILABLE:
        chat_tab, pdf_tab, graph_tab = st.tabs(["💬 Chat with PDF", "📄 View PDF", "🕸️ Knowledge Graph"])
    else:
        chat_tab, pdf_tab = st.tabs(["💬 Chat with PDF", "📄 View PDF"])
        graph_tab = None

    with pdf_tab:
        st.markdown("### PDF Document")

        if st.session_state.get('document_topics'):
            st.markdown("**🏷️ Тематические метки:**")
            topics_html = ""
            for topic in st.session_state.document_topics:
                topics_html += f'<span style="background-color: #e1f5fe; color: #01579b; padding: 2px 8px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 0.85em;">{topic}</span>'
            st.markdown(topics_html, unsafe_allow_html=True)
            st.markdown("")

        if st.session_state.get('document_summary'):
            st.markdown("**📋 Краткое описание:**")
            st.info(st.session_state.document_summary)

        if st.session_state.get('document_images'):
            with st.expander(f"🖼️ Изображения в документе ({len(st.session_state.document_images)})", expanded=False):
                for i, img in enumerate(st.session_state.document_images[:5]):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        try:
                            img_data = base64.b64decode(img['data'])
                            st.image(img_data, width=100, caption=f"Страница {img['page']}")
                        except:
                            st.write(f"📄 Страница {img['page']}")
                    with col2:
                        st.write(f"**Размер:** {img['width']}×{img['height']} пикселей")
                        st.write(f"**Размер файла:** {img['size_bytes']:,} байт")

        if st.session_state.get('pdf_base64'):
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

    if graph_tab is not None:
        with graph_tab:
            render_knowledge_graph()

    with chat_tab:
        st.markdown("### Chat Interface")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

if st.session_state.pdf_processed and api_configured:
    if prompt := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    if st.session_state.lightrag_processor is not None:
                        st.info("🧠 Querying knowledge graph...")
                        proc = st.session_state.lightrag_processor
                        proc.last_llm_error = None
                        
                        response = run_async_query(
                            proc,
                            prompt,
                            mode="local",
                            top_k=5,
                            response_type="comprehensive"
                        )

                        if proc.last_llm_error:
                            st.error(f"⚠️ LLM API error during query: {proc.last_llm_error}")

                        if st.session_state.get('debug_mode', False):
                            storage_stats = proc.get_storage_stats()
                            st.info(f"🧠 **Knowledge Graph Stats:** {storage_stats.get('file_count', 0)} storage files")

                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    elif st.session_state.vector_store is not None:
                        relevant_chunks = st.session_state.vector_store.search(prompt, k=5, score_threshold=0.5)

                        if st.session_state.get('debug_mode', False):
                            stats = st.session_state.vector_store.get_stats()
                            total_chunks = stats.get('total_chunks', 0)
                            st.info(f"🔍 **Search Results:** Found {len(relevant_chunks)} relevant chunks from {total_chunks} total chunks")

                            if relevant_chunks:
                                with st.expander("📄 View Retrieved Context", expanded=False):
                                    for i, chunk in enumerate(relevant_chunks[:3]):
                                        st.write(f"**Chunk {i+1}** (Score: {chunk.get('score', 0):.3f})")
                                        st.write(chunk.get('content', '')[:300] + "...")
                                        st.divider()

                        if relevant_chunks:
                            context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
                            openrouter_client = OpenRouterClient(api_key, base_url)

                            response = openrouter_client.get_response(
                                messages=st.session_state.messages[:-1],
                                question=prompt,
                                context=context,
                                model=st.session_state.selected_model,
                                max_tokens=1000,
                                temperature=0.7
                            )

                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            st.warning("No relevant content found in the document.")
                    else:
                        st.error("❌ No processing method available. Please re-upload the document.")

                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Built with Streamlit | Powered by OpenRouter API | Vector search with scikit-learn
    </div>
    """,
    unsafe_allow_html=True
)
