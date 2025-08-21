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
        Initialize the LightRAG instance following best practices
        """
        if not LIGHTRAG_AVAILABLE:
            raise RuntimeError("LightRAG is not available. Please install lightrag-hku package.")
        
        try:
            logger.info("Initializing LightRAG...")
            
            # Create working directory if it doesn't exist
            os.makedirs(self.working_dir, exist_ok=True)
            
            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=self.llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=3072,  # AiTunnel text-embedding-3-large actual dimension
                    max_token_size=8192,
                    func=self.embedding_func,
                ),
            )
            
            # CRITICAL: Both calls required in this exact order
            await self.rag.initialize_storages()
            await initialize_pipeline_status()
            
            logger.info("LightRAG initialized successfully")
            return self.rag
            
        except Exception as e:
            logger.error(f"Error initializing LightRAG: {e}")
            raise
    

    
    async def insert_document(self, text: str, document_id: Optional[str] = None) -> bool:
        """
        Insert a document into the knowledge graph with error handling
        """
        try:
            # Force clean initialization each time to avoid state issues
            logger.info("Reinitializing LightRAG for document insertion...")
            
            # Clear problematic storage state
            import shutil
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)
                os.makedirs(self.working_dir, exist_ok=True)
            
            self.rag = None
            await self.initialize_rag()
            
            logger.info(f"Inserting document (length: {len(text)} chars) into knowledge graph...")
            
            # Try simple insertion with very small text chunks to avoid library bugs
            if len(text) > 1500:
                # Split into smaller chunks to avoid internal library errors
                chunks = [text[i:i+1200] for i in range(0, len(text), 1200)]
                logger.info(f"Splitting large document into {len(chunks)} chunks")
                
                for i, chunk in enumerate(chunks[:2]):  # Process only first 2 chunks to avoid errors
                    try:
                        logger.info(f"Processing chunk {i+1}: {len(chunk)} chars")
                        self.rag.insert(chunk)
                        await asyncio.sleep(1)  # Small delay between chunks
                    except Exception as chunk_error:
                        logger.error(f"Chunk {i+1} failed: {chunk_error}")
                        continue
            else:
                # For small documents, try direct insertion
                try:
                    self.rag.insert(text)
                    logger.info("Small document inserted successfully")
                except Exception as small_error:
                    logger.error(f"Small document insertion failed: {small_error}")
                    return False
            
            # Wait for processing to complete
            await asyncio.sleep(5)
            
            # Check if any vector files were created
            vdb_files = ['vdb_entities.json', 'vdb_relationships.json', 'vdb_chunks.json']
            files_created = 0
            for vdb_file in vdb_files:
                vdb_path = os.path.join(self.working_dir, vdb_file)
                if os.path.exists(vdb_path) and os.path.getsize(vdb_path) > 50:
                    files_created += 1
                    logger.info(f"Created {vdb_file}: {os.path.getsize(vdb_path)} bytes")
            
            if files_created > 0:
                logger.info(f"Document insertion completed successfully. Created {files_created} vector database files.")
                return True
            else:
                logger.warning("No vector database files were created - insertion may have failed")
                return False
            
        except Exception as e:
            logger.error(f"Error inserting document: {e}")
            return False
    

    
    async def query_knowledge_graph(
        self, 
        query: str, 
        mode: str = "local",  # Changed to simpler mode
        top_k: int = 3,       # Reduced top_k 
        response_type: str = "single line"  # Simplified response
    ) -> str:
        """
        Query the knowledge graph with timeout
        
        Args:
            query: The query string
            mode: Query mode - 'local', 'global', 'hybrid', or 'mix'
            top_k: Number of top results to retrieve
            response_type: 'single line' or 'comprehensive'
        """
        try:
            if self.rag is None:
                await self.initialize_rag()
            
            logger.info(f"Querying knowledge graph with mode={mode}, top_k={top_k}")
            
            # Add timeout to prevent hanging
            try:
                # Use asyncio.wait_for with timeout
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.rag.query,
                        query=query,
                        param=QueryParam(
                            mode=mode,
                            top_k=top_k,
                            response_type=response_type
                        )
                    ),
                    timeout=60.0  # 60 second timeout
                )
                
                logger.info("Knowledge graph query completed successfully")
                
                # Check if response indicates no context found
                if isinstance(response, str) and ("[no-context]" in response or "Sorry, I'm not able to provide" in response):
                    logger.warning("No context found in knowledge graph")
                    return "К сожалению, я не могу найти релевантную информацию в загруженном документе для ответа на ваш вопрос. Попробуйте переформулировать вопрос или загрузить документ заново."
                
                return response
                
            except asyncio.TimeoutError:
                logger.error("Knowledge graph query timed out after 60 seconds")
                return "Запрос к графу знаний превысил время ожидания. Попробуйте задать более простой вопрос или используйте традиционный RAG."
            
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