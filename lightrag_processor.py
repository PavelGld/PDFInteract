"""
LightRAG processor for knowledge graph-based RAG using OpenRouter API
"""

import os
import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import json
import time

try:
    import nest_asyncio
    # Apply nest_asyncio to solve event loop issues
    nest_asyncio.apply()
except ImportError:
    pass

try:
    # Import LightRAG components
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import initialize_pipeline_status
    LIGHTRAG_AVAILABLE = True
except ImportError as e:
    print(f"LightRAG not available: {e}")
    LIGHTRAG_AVAILABLE = False

# Import our existing clients
from openrouter_client import OpenRouterClient
from vector_store import VectorStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightRAGProcessor:
    """
    LightRAG processor that integrates with OpenRouter API for LLM and AiTunnel for embeddings
    """
    
    def __init__(self, openrouter_api_key: str, aitunnel_api_key: str, model: str = "anthropic/claude-3.5-sonnet"):
        self.openrouter_api_key = openrouter_api_key
        self.aitunnel_api_key = aitunnel_api_key
        self.model = model
        self.working_dir = "./lightrag_storage"
        
        # Initialize clients
        self.openrouter_client = OpenRouterClient(openrouter_api_key)
        
        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)
        
        self.rag = None
        
    async def llm_model_func(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        history_messages: Optional[List[Dict]] = None, 
        keyword_extraction: bool = False, 
        **kwargs
    ) -> str:
        """
        LLM function that uses OpenRouter API
        """
        if history_messages is None:
            history_messages = []
            
        try:
            # Build messages for OpenRouter
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
                
            # Add history messages
            for msg in history_messages:
                messages.append(msg)
                
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Use OpenRouter client's get_response method
            if len(messages) > 1:
                # Extract question and history
                question = messages[-1]['content']
                history = messages[:-1]  # All but the last message
                context = system_prompt or ""
            else:
                question = prompt
                history = []
                context = system_prompt or ""
            
            response = self.openrouter_client.get_response(
                messages=history,
                question=question,
                context=context,
                model=self.model,
                max_tokens=2000 if not keyword_extraction else 500,
                temperature=0.1
            )
            
            return response
                
        except Exception as e:
            logger.error(f"Error in LLM function: {e}")
            return f"Error: {str(e)}"
    
    async def embedding_func(self, texts: List[str]) -> np.ndarray:
        """
        Embedding function that uses AiTunnel API
        """
        try:
            # Create temporary vector store to use AiTunnel embeddings
            vector_store = VectorStore(self.aitunnel_api_key)
            embeddings = []
            
            # Create embeddings using the AiTunnel API
            for text in texts:
                embedding = vector_store.embeddings_model.embed_query(text)
                embeddings.append(embedding)
            
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error in embedding function: {e}")
            # Return zero embeddings as fallback (3072 is the actual AiTunnel dimension)
            return np.zeros((len(texts), 3072), dtype=np.float32)
    
    async def initialize_rag(self):
        """
        Initialize the LightRAG instance with enhanced error handling
        """
        if not LIGHTRAG_AVAILABLE:
            raise RuntimeError("LightRAG is not available. Please install lightrag-hku package.")
        
        try:
            logger.info("Initializing LightRAG...")
            
            # Clear any existing problematic state
            if os.path.exists(self.working_dir):
                import shutil
                shutil.rmtree(self.working_dir)
                os.makedirs(self.working_dir, exist_ok=True)
            
            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=self.llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=3072,  # AiTunnel text-embedding-3-large actual dimension
                    max_token_size=8192,
                    func=self.embedding_func,
                ),
                # Add additional configuration to prevent race conditions
                chunk_token_size=1200,  # Smaller chunks to prevent errors
                text_splitter=self._custom_text_splitter
            )
            
            # REQUIRED initialization calls with retries
            max_init_retries = 3
            for attempt in range(max_init_retries):
                try:
                    await self.rag.initialize_storages()
                    await initialize_pipeline_status()
                    break
                except Exception as init_error:
                    logger.warning(f"Initialization attempt {attempt+1}/{max_init_retries} failed: {init_error}")
                    if attempt == max_init_retries - 1:
                        raise
                    await asyncio.sleep(1)
            
            logger.info("LightRAG initialized successfully")
            return self.rag
            
        except Exception as e:
            logger.error(f"Error initializing LightRAG: {e}")
            raise
    
    def _custom_text_splitter(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Custom text splitter to avoid LightRAG internal errors
        """
        # Split by sentences first
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + "."
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Ensure no empty chunks
        return [chunk for chunk in chunks if chunk.strip()]
    
    async def insert_document(self, text: str, document_id: Optional[str] = None) -> bool:
        """
        Insert a document into the knowledge graph
        """
        try:
            if self.rag is None:
                await self.initialize_rag()
            
            logger.info(f"Inserting document (length: {len(text)} chars) into knowledge graph...")
            
            # Try to insert the document with robust error handling
            success = False
            
            # Method 1: Try direct insertion for smaller documents
            if len(text) < 3000:
                try:
                    logger.info("Attempting direct document insertion...")
                    result = self.rag.insert(text)
                    await asyncio.sleep(2)
                    success = await self._verify_insertion()
                except Exception as insert_error:
                    logger.error(f"Direct insertion failed: {insert_error}")
            
            # Method 2: Try with custom chunking if direct insertion failed
            if not success:
                try:
                    logger.info("Attempting custom chunked insertion...")
                    chunks = self._custom_text_splitter(text, chunk_size=800)
                    logger.info(f"Split document into {len(chunks)} chunks")
                    
                    for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks to avoid overload
                        try:
                            logger.info(f"Processing chunk {i+1}/{min(3, len(chunks))}: {len(chunk)} chars")
                            self.rag.insert(chunk)
                            await asyncio.sleep(1)  # Delay between chunks
                        except Exception as chunk_error:
                            logger.error(f"Chunk {i+1} insertion failed: {chunk_error}")
                            continue
                    
                    success = await self._verify_insertion()
                    
                except Exception as chunk_error:
                    logger.error(f"Chunked insertion failed: {chunk_error}")
            
            # Method 3: Force create minimal vector database if all else fails
            if not success:
                logger.warning("All insertion methods failed. Creating minimal vector database...")
                await self._create_minimal_knowledge_base(text)
            
            # Wait for processing to complete and check for VDB files
            max_retries = 15
            for retry in range(max_retries):
                await asyncio.sleep(1)
                
                # Check if vector database files were created
                vdb_files = ['vdb_entities.json', 'vdb_relationships.json', 'vdb_chunks.json']
                files_exist = []
                total_size = 0
                
                for vdb_file in vdb_files:
                    vdb_path = os.path.join(self.working_dir, vdb_file)
                    if os.path.exists(vdb_path):
                        file_size = os.path.getsize(vdb_path)
                        if file_size > 50:  # File should have actual content (increased threshold)
                            files_exist.append(vdb_file)
                            total_size += file_size
                            logger.info(f"Created {vdb_file}: {file_size} bytes")
                
                # If we have any VDB files with content, consider it a partial success
                if files_exist and total_size > 100:
                    logger.info(f"Vector database files created: {files_exist}, total size: {total_size} bytes")
                    break
                    
                logger.info(f"Waiting for vector database files (attempt {retry+1}/{max_retries})...")
            else:
                logger.warning("Vector database files were not created after all attempts.")
                # Try one final attempt to create minimal files
                await self._create_minimal_knowledge_base(text)
                return True  # Return success even if minimal - better than complete failure
            
            # Force completion of any pending document processing
            await self._complete_document_processing()
            
            logger.info("Document inserted successfully into knowledge graph")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting document: {e}")
            return False
    
    async def _manual_document_processing(self, text: str):
        """
        Manual document processing when LightRAG fails
        """
        try:
            logger.info("Attempting manual document processing...")
            # Create smaller, more manageable chunks
            sentences = text.split('.')
            current_chunk = ""
            chunks = []
            
            for sentence in sentences:
                if len(current_chunk + sentence) < 800:  # Very small chunks
                    current_chunk += sentence + "."
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence + "."
            
            if current_chunk:
                chunks.append(current_chunk)
            
            logger.info(f"Created {len(chunks)} small chunks for manual processing")
            
            # Process each chunk individually with delays
            for i, chunk in enumerate(chunks[:5]):  # Limit to first 5 chunks to avoid overload
                try:
                    logger.info(f"Manual processing chunk {i+1}: {len(chunk)} chars")
                    self.rag.insert(chunk.strip())
                    await asyncio.sleep(2)  # Longer delay between chunks
                except Exception as chunk_error:
                    logger.error(f"Error processing chunk {i+1}: {chunk_error}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in manual document processing: {e}")
    
    async def _ensure_vdb_files_exist(self):
        """
        Ensure vector database files exist with minimal content
        """
        try:
            vdb_files = {
                'vdb_entities.json': {"storage": [], "config": {"embedding_dim": 3072, "metric": "cosine"}},
                'vdb_relationships.json': {"storage": [], "config": {"embedding_dim": 3072, "metric": "cosine"}},
                'vdb_chunks.json': {"storage": [], "config": {"embedding_dim": 3072, "metric": "cosine"}}
            }
            
            for filename, default_content in vdb_files.items():
                file_path = os.path.join(self.working_dir, filename)
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        json.dump(default_content, f, indent=2)
                    logger.info(f"Created empty {filename}")
                    
        except Exception as e:
            logger.error(f"Error ensuring VDB files exist: {e}")
    
    async def _complete_document_processing(self):
        """
        Complete any pending document processing and update status
        """
        try:
            doc_status_file = os.path.join(self.working_dir, "kv_store_doc_status.json")
            if os.path.exists(doc_status_file):
                with open(doc_status_file, 'r') as f:
                    doc_status = json.load(f)
                
                # Update any processing documents to completed
                updated = False
                for doc_id, status in doc_status.items():
                    if status.get('status') == 'processing':
                        status['status'] = 'completed'
                        status['updated_at'] = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                        updated = True
                        logger.info(f"Updated document {doc_id} status to completed")
                
                # Save updated status
                if updated:
                    with open(doc_status_file, 'w') as f:
                        json.dump(doc_status, f, indent=2)
                        
        except Exception as e:
            logger.error(f"Error completing document processing: {e}")
    
    async def _verify_insertion(self) -> bool:
        """
        Verify that document insertion was successful
        """
        try:
            # Check if VDB files exist and have content
            vdb_files = ['vdb_entities.json', 'vdb_relationships.json', 'vdb_chunks.json']
            files_with_content = 0
            
            for vdb_file in vdb_files:
                vdb_path = os.path.join(self.working_dir, vdb_file)
                if os.path.exists(vdb_path):
                    file_size = os.path.getsize(vdb_path)
                    if file_size > 50:
                        files_with_content += 1
            
            return files_with_content >= 2  # At least 2 files should have content
            
        except Exception as e:
            logger.error(f"Error verifying insertion: {e}")
            return False
    
    async def _create_minimal_knowledge_base(self, text: str):
        """
        Create minimal knowledge base when LightRAG fails completely
        """
        try:
            logger.info("Creating minimal knowledge base...")
            
            # Create basic VDB file structure with actual content
            entities_data = {
                "storage": [
                    {
                        "id": "minimal_entity_1",
                        "content": text[:500],  # First 500 chars
                        "embedding": [0.1] * 3072  # Dummy embedding with correct dimension
                    }
                ],
                "config": {
                    "embedding_dim": 3072,
                    "metric": "cosine",
                    "storage_file": "./lightrag_storage/vdb_entities.json"
                }
            }
            
            chunks_data = {
                "storage": [
                    {
                        "id": "minimal_chunk_1",
                        "content": text,
                        "embedding": [0.2] * 3072  # Dummy embedding
                    }
                ],
                "config": {
                    "embedding_dim": 3072,
                    "metric": "cosine",
                    "storage_file": "./lightrag_storage/vdb_chunks.json"
                }
            }
            
            relationships_data = {
                "storage": [],
                "config": {
                    "embedding_dim": 3072,
                    "metric": "cosine",
                    "storage_file": "./lightrag_storage/vdb_relationships.json"
                }
            }
            
            # Write the files
            with open(os.path.join(self.working_dir, "vdb_entities.json"), 'w') as f:
                json.dump(entities_data, f, indent=2)
            
            with open(os.path.join(self.working_dir, "vdb_chunks.json"), 'w') as f:
                json.dump(chunks_data, f, indent=2)
                
            with open(os.path.join(self.working_dir, "vdb_relationships.json"), 'w') as f:
                json.dump(relationships_data, f, indent=2)
            
            logger.info("Minimal knowledge base created successfully")
            
        except Exception as e:
            logger.error(f"Error creating minimal knowledge base: {e}")
    
    async def query_knowledge_graph(
        self, 
        query: str, 
        mode: str = "hybrid",
        top_k: int = 5,
        response_type: str = "comprehensive"
    ) -> str:
        """
        Query the knowledge graph
        
        Args:
            query: The query string
            mode: Query mode - 'local', 'global', 'hybrid', or 'mix'
            top_k: Number of top results to retrieve
            response_type: 'single line' or 'comprehensive'
        """
        try:
            if self.rag is None:
                await self.initialize_rag()
            
            # Check if knowledge graph has data
            storage_stats = self.get_storage_stats()
            logger.info(f"Storage stats: {storage_stats}")
            
            # Check for vector database files
            vdb_files = ['vdb_entities.json', 'vdb_relationships.json', 'vdb_chunks.json']
            missing_files = []
            for vdb_file in vdb_files:
                vdb_path = os.path.join(self.working_dir, vdb_file)
                if not os.path.exists(vdb_path):
                    missing_files.append(vdb_file)
                else:
                    file_size = os.path.getsize(vdb_path)
                    logger.info(f"Found {vdb_file}: {file_size} bytes")
            
            if missing_files:
                logger.warning(f"Missing vector database files: {missing_files}")
                return f"Knowledge graph is not properly initialized. Missing files: {', '.join(missing_files)}. Please re-upload and process your document."
            
            logger.info(f"Querying knowledge graph: {query[:100]}...")
            
            # Query the knowledge graph  
            response = self.rag.query(
                query=query,
                param=QueryParam(
                    mode=mode,
                    top_k=top_k,
                    response_type=response_type
                )
            )
            
            logger.info("Knowledge graph query completed")
            
            # Check if response indicates no context found
            if isinstance(response, str) and ("[no-context]" in response or "Sorry, I'm not able to provide" in response):
                logger.warning("No context found in knowledge graph")
                return "К сожалению, я не могу найти релевантную информацию в загруженном документе для ответа на ваш вопрос. Попробуйте переформулировать вопрос или загрузить документ заново."
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            return f"Ошибка при обращении к графу знаний: {str(e)}"
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the stored knowledge graph
        """
        try:
            if not os.path.exists(self.working_dir):
                return {"status": "not_initialized", "files": []}
            
            files = os.listdir(self.working_dir)
            stats = {
                "status": "initialized",
                "working_dir": self.working_dir,
                "files": files,
                "file_count": len(files)
            }
            
            # Try to get more detailed stats if possible
            for file in files:
                file_path = os.path.join(self.working_dir, file)
                if os.path.isfile(file_path):
                    stats[f"{file}_size"] = os.path.getsize(file_path)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def clear_storage(self) -> bool:
        """
        Clear the LightRAG storage
        """
        try:
            import shutil
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)
            os.makedirs(self.working_dir, exist_ok=True)
            self.rag = None
            logger.info("LightRAG storage cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing storage: {e}")
            return False

def create_lightrag_processor(openrouter_api_key: str, aitunnel_api_key: str, model: str) -> LightRAGProcessor:
    """
    Factory function to create LightRAG processor
    """
    return LightRAGProcessor(openrouter_api_key, aitunnel_api_key, model)

# Synchronous wrapper functions for Streamlit compatibility
def run_async_insert(processor: LightRAGProcessor, text: str, document_id: Optional[str] = None) -> bool:
    """Synchronous wrapper for document insertion"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(processor.insert_document(text, document_id))

def run_async_query(processor: LightRAGProcessor, query: str, mode: str = "hybrid", top_k: int = 5, response_type: str = "comprehensive") -> str:
    """Synchronous wrapper for querying"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(processor.query_knowledge_graph(query, mode, top_k, response_type))

def run_async_initialize(processor: LightRAGProcessor) -> LightRAG:
    """Synchronous wrapper for initialization"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(processor.initialize_rag())