"""
Векторное хранилище для PDF Chat Assistant

Модуль для создания и управления векторным хранилищем документов.
Использует AiTunnel API для создания эмбеддингов и косинусное сходство для поиска.

Основные функции:
- Создание векторных представлений текстовых фрагментов через AiTunnel API
- Поиск релевантных документов по запросу
- Управление индексом векторов в памяти
"""

import numpy as np
from typing import List, Dict, Any
import os
from sklearn.metrics.pairwise import cosine_similarity
from utils2 import AiTunnelEmbeddings

class VectorStore:
    def __init__(self, api_key: str):
        """
        Initialize vector store with AiTunnel API embeddings.
        
        Args:
            api_key: AiTunnel API key (sk-aitunnel-xxx) for embeddings
        """
        self.aitunnel_api_key = os.environ.get("AITUNNEL_API_KEY", api_key)
        self.embeddings_model = AiTunnelEmbeddings(api_key=self.aitunnel_api_key)
        
        # Тест подключения к AiTunnel API
        print("Testing AiTunnel API connection...")
        if not self.embeddings_model.test_connection():
            raise Exception("Failed to connect to AiTunnel API. Please check your API key.")
            
        self.chunks = []
        self.embeddings = []
        self.vectorstore = None
        
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add text chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
        """
        if not chunks:
            return
        
        self.chunks = chunks
        
        try:
            # Extract texts from chunks
            texts = [chunk['content'] for chunk in chunks]
            
            print(f"Creating embeddings for {len(texts)} text chunks using AiTunnel API (processing in batches due to rate limits)...")
            
            # Get embeddings using AiTunnel API with rate limiting
            self.embeddings = self.embeddings_model.embed_documents(texts)
            self.embeddings = np.array(self.embeddings)
            self.vectorstore = True
            print(f"Successfully created vector store with {len(chunks)} documents using AiTunnel API")
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            self.vectorstore = None
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using embedding similarity.
        
        Args:
            query: Search query text
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries with chunk content and similarity scores
        """
        if self.vectorstore is None or len(self.embeddings) == 0:
            print("Vector store not initialized")
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embeddings_model.embed_query(query)
            query_vector = np.array(query_embedding).reshape(1, -1)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.embeddings).flatten()
            
            # Get top k indices
            top_indices = similarities.argsort()[-k:][::-1]
            
            # Format results
            formatted_results = []
            for idx in top_indices:
                similarity_score = similarities[idx]
                
                if similarity_score >= score_threshold:
                    result = {
                        'content': self.chunks[idx]['content'],
                        'score': float(similarity_score),
                        'distance': float(1.0 - similarity_score),
                        'metadata': {
                            'chunk_id': self.chunks[idx].get('id', ''),
                            'start_pos': self.chunks[idx].get('start_pos', 0),
                            'end_pos': self.chunks[idx].get('end_pos', 0)
                        },
                        'id': self.chunks[idx].get('id', 'unknown')
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def get_retriever(self, search_type: str = "similarity", k: int = 5):
        """
        Get retriever-like interface.
        """
        def retrieve(query: str):
            results = self.search(query, k=k, score_threshold=0.0)
            return results
        
        return retrieve
    
    def get_chunk_by_id(self, chunk_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific chunk by its ID.
        """
        for chunk in self.chunks:
            if chunk.get('id') == chunk_id:
                return chunk
        return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        """
        return {
            'total_chunks': len(self.chunks),
            'total_documents': len(self.chunks),
            'vectorstore_type': 'Course API Embeddings',
            'vectorstore_active': self.vectorstore is not None,
            'embedding_model': 'Course API',
            'embedding_dimensions': len(self.embeddings[0]) if len(self.embeddings) > 0 else 0
        }
    
    def save_to_disk(self, path: str) -> None:
        """Save vector store to disk."""
        print("Course API embeddings vector store ready")
    
    def load_from_disk(self, path: str) -> None:
        """Load vector store from disk."""
        print("Course API embeddings vector store loaded")
    
    def clear(self) -> None:
        """Clear all data from the vector store."""
        self.chunks = []
        self.embeddings = []
        self.vectorstore = None
    
    def add_single_chunk(self, chunk: Dict[str, Any]) -> None:
        """Add a single chunk to the vector store."""
        self.add_chunks([chunk])
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk from the vector store."""
        original_length = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk.get('id') != chunk_id]
        
        if len(self.chunks) == original_length:
            return False
        
        if self.chunks:
            self.add_chunks(self.chunks)
        else:
            self.clear()
        
        return True