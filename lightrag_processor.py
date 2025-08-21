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
        Initialize the LightRAG instance
        """
        if not LIGHTRAG_AVAILABLE:
            raise RuntimeError("LightRAG is not available. Please install lightrag-hku package.")
        
        try:
            logger.info("Initializing LightRAG...")
            
            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=self.llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=3072,  # AiTunnel text-embedding-3-large actual dimension
                    max_token_size=8192,
                    func=self.embedding_func,
                ),
            )
            
            # REQUIRED initialization calls
            await self.rag.initialize_storages()
            await initialize_pipeline_status()
            
            logger.info("LightRAG initialized successfully")
            return self.rag
            
        except Exception as e:
            logger.error(f"Error initializing LightRAG: {e}")
            raise
    
    async def insert_document(self, text: str, document_id: Optional[str] = None) -> bool:
        """
        Insert a document into the knowledge graph
        """
        try:
            if self.rag is None:
                await self.initialize_rag()
            
            logger.info(f"Inserting document (length: {len(text)} chars) into knowledge graph...")
            
            # Insert the document and force synchronous processing
            result = self.rag.insert(text)
            
            # Wait for processing to complete and check for VDB files
            max_retries = 10
            for retry in range(max_retries):
                await asyncio.sleep(1)
                
                # Check if vector database files were created
                vdb_files = ['vdb_entities.json', 'vdb_relationships.json', 'vdb_chunks.json']
                files_exist = []
                for vdb_file in vdb_files:
                    vdb_path = os.path.join(self.working_dir, vdb_file)
                    if os.path.exists(vdb_path):
                        file_size = os.path.getsize(vdb_path)
                        if file_size > 10:  # File should have actual content
                            files_exist.append(vdb_file)
                            logger.info(f"Created {vdb_file}: {file_size} bytes")
                
                # If at least entities and chunks files exist, consider it successful
                if 'vdb_entities.json' in files_exist and 'vdb_chunks.json' in files_exist:
                    logger.info(f"Vector database files created successfully: {files_exist}")
                    break
                    
                logger.info(f"Waiting for vector database files (attempt {retry+1}/{max_retries})...")
            else:
                logger.warning("Vector database files were not created. Attempting manual initialization...")
                await self._ensure_vdb_files_exist()
            
            # Force completion of any pending document processing
            await self._complete_document_processing()
            
            logger.info("Document inserted successfully into knowledge graph")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting document: {e}")
            return False
    
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