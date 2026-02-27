"""
LightRAG processor for knowledge graph-based RAG using user-provided API credentials
"""

import os
os.environ["TIMEOUT"] = "600"
os.environ["READ_TIMEOUT"] = "600"
os.environ["HTTPX_TIMEOUT"] = "600"
os.environ["LLM_REQUEST_TIMEOUT"] = "600"

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import json
import time

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import initialize_pipeline_status
    LIGHTRAG_AVAILABLE = True
except ImportError as e:
    print(f"LightRAG not available: {e}")
    LIGHTRAG_AVAILABLE = False

from openrouter_client import OpenRouterClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightRAGProcessor:
    """
    LightRAG processor that uses user-provided API credentials for LLM and embeddings.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        embeddings_api_key: str = "",
        embeddings_base_url: str = "https://api.aitunnel.ru/v1/",
        model: str = "openai/gpt-4o",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.embeddings_api_key = embeddings_api_key or api_key
        self.embeddings_base_url = embeddings_base_url
        self.model = model
        self.working_dir = "./lightrag_storage"

        self.openrouter_client = OpenRouterClient(api_key, base_url)

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
        if history_messages is None:
            history_messages = []

        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            for msg in history_messages:
                messages.append(msg)

            messages.append({"role": "user", "content": prompt})

            if len(messages) > 1:
                question = messages[-1]['content']
                history = messages[:-1]
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
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.embeddings_api_key,
                base_url=self.embeddings_base_url,
            )

            embeddings = []
            for text in texts:
                resp = client.embeddings.create(
                    input=text,
                    model="text-embedding-3-large"
                )
                embeddings.append(resp.data[0].embedding)

            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error in embedding function: {e}")
            return np.zeros((len(texts), 3072), dtype=np.float32)

    async def initialize_rag(self):
        if not LIGHTRAG_AVAILABLE:
            raise RuntimeError("LightRAG is not available. Please install lightrag-hku package.")

        try:
            try:
                from lightrag_timeout_fix import patch_lightrag_timeouts, setup_environment_timeouts
                patch_lightrag_timeouts()
                setup_environment_timeouts()
                logger.info("Applied LightRAG timeout patches")
            except ImportError:
                logger.info("Timeout fix module not available, using environment variables only")

            logger.info("Initializing LightRAG...")

            os.makedirs(self.working_dir, exist_ok=True)

            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=self.llm_model_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=3072,
                    max_token_size=8192,
                    func=self.embedding_func,
                ),
            )

            await self.rag.initialize_storages()
            await initialize_pipeline_status()

            logger.info("LightRAG initialized successfully")
            return self.rag

        except Exception as e:
            logger.error(f"Error initializing LightRAG: {e}")
            raise

    async def insert_document(self, text: str, document_id: Optional[str] = None) -> bool:
        try:
            if self.rag is None:
                await self.initialize_rag()

            logger.info(f"Inserting document (length: {len(text)} chars) into knowledge graph...")

            try:
                logger.info("Attempting document insertion...")
                self.rag.insert(text)
                logger.info("Document inserted successfully")
            except Exception as insert_error:
                logger.error(f"Document insertion failed: {insert_error}")
                return False

            await asyncio.sleep(5)

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
        mode: str = "local",
        top_k: int = 3,
        response_type: str = "single line"
    ) -> str:
        try:
            if self.rag is None:
                await self.initialize_rag()

            logger.info(f"Querying knowledge graph with mode={mode}, top_k={top_k}")
            logger.info(f"Query: {query[:100]}...")

            storage_files = os.listdir(self.working_dir)
            logger.info(f"Storage files available: {storage_files}")

            try:
                logger.info("Calling self.rag.query()...")

                debug_params = QueryParam(
                    mode=mode,
                    top_k=top_k,
                    response_type=response_type,
                    enable_rerank=False
                )

                start_time = time.time()
                response = self.rag.query(query=query, param=debug_params)
                duration = time.time() - start_time
                logger.info(f"Query completed in {duration:.2f} seconds")

            except Exception as query_exception:
                logger.error(f"Query failed with exception: {query_exception}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                raise

            if isinstance(response, str) and ("[no-context]" in response or "Sorry, I'm not able to provide" in response):
                logger.warning("No context found in knowledge graph")
                return "К сожалению, я не могу найти релевантную информацию в загруженном документе для ответа на ваш вопрос. Попробуйте переформулировать вопрос или загрузить документ заново."

            return response

        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            return f"Ошибка при обращении к графу знаний: {str(e)}"

    def get_storage_stats(self) -> Dict[str, Any]:
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

            for file in files:
                file_path = os.path.join(self.working_dir, file)
                if os.path.isfile(file_path):
                    stats[f"{file}_size"] = os.path.getsize(file_path)

            return stats

        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {"status": "error", "error": str(e)}

    def clear_storage(self) -> bool:
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


def create_lightrag_processor(
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    embeddings_api_key: str = "",
    embeddings_base_url: str = "https://api.aitunnel.ru/v1/",
    model: str = "openai/gpt-4o",
) -> LightRAGProcessor:
    return LightRAGProcessor(
        api_key=api_key,
        base_url=base_url,
        embeddings_api_key=embeddings_api_key,
        embeddings_base_url=embeddings_base_url,
        model=model,
    )


def run_async_insert(processor: LightRAGProcessor, text: str, document_id: Optional[str] = None) -> bool:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(processor.insert_document(text, document_id))

def run_async_query(processor: LightRAGProcessor, query: str, mode: str = "hybrid", top_k: int = 5, response_type: str = "comprehensive") -> str:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(processor.query_knowledge_graph(query, mode, top_k, response_type))

def run_async_initialize(processor: LightRAGProcessor) -> LightRAG:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(processor.initialize_rag())
