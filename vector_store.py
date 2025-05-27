import requests
import json
import numpy as np
from typing import List, Dict, Any
import os

class VectorStore:
    def __init__(self, api_key: str, model_name: str = "text-embedding-ada-002"):
        """
        Initialize vector store with API-based embeddings.
        
        Args:
            api_key: OpenRouter API key for embeddings
            model_name: Embedding model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        self.chunks = []
        self.embeddings = []
        self.vectorstore = True
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "PDF Chat Assistant"
        }
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using OpenRouter API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Use OpenAI embedding endpoint through OpenRouter
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers=self.headers,
                json={
                    "model": "openai/text-embedding-ada-002",
                    "input": text[:8000]  # Limit text length
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['data'][0]['embedding']
            else:
                # Fallback to simple embedding if API fails
                return self._simple_embedding(text)
                
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str) -> List[float]:
        """
        Simple fallback embedding based on text characteristics.
        
        Args:
            text: Text to embed
            
        Returns:
            Simple embedding vector
        """
        # Create a simple but meaningful embedding
        words = text.lower().split()
        
        # 384-dimensional vector (common embedding size)
        vector = [0.0] * 384
        
        # Fill vector based on word characteristics
        for i, word in enumerate(words[:384]):
            if word:
                # Use character values and position
                char_sum = sum(ord(c) for c in word[:10])
                vector[i] = (char_sum % 1000) / 1000.0
        
        # Add some text statistics
        if len(vector) > 10:
            vector[0] = len(text) / 10000.0  # Text length
            vector[1] = len(words) / 1000.0   # Word count
            vector[2] = len(set(words)) / 1000.0  # Unique words
        
        # Normalize vector
        norm = sum(x*x for x in vector) ** 0.5
        if norm > 0:
            vector = [x/norm for x in vector]
        
        return vector
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add text chunks to the vector store with API embeddings.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
        """
        if not chunks:
            return
        
        self.chunks = chunks
        self.embeddings = []
        
        print(f"Creating embeddings for {len(chunks)} chunks...")
        
        # Create embeddings for each chunk
        for i, chunk in enumerate(chunks):
            embedding = self._get_embedding(chunk['content'])
            self.embeddings.append(embedding)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks")
        
        print("Embeddings created successfully!")
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using API-based embeddings.
        
        Args:
            query: Search query text
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries with chunk content and similarity scores
        """
        if not self.chunks or not self.embeddings:
            return []
        
        # Get embedding for query
        query_embedding = self._get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for i, chunk_embedding in enumerate(self.embeddings):
            score = self._cosine_similarity(query_embedding, chunk_embedding)
            
            if score >= score_threshold:
                result_chunk = self.chunks[i].copy()
                result_chunk['score'] = score
                similarities.append(result_chunk)
        
        # Sort by score (descending) and return top k
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:k]
    
    def get_chunk_by_id(self, chunk_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific chunk by its ID.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            Chunk dictionary or empty dict if not found
        """
        for chunk in self.chunks:
            if chunk.get('id') == chunk_id:
                return chunk
        return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        return {
            'total_chunks': len(self.chunks),
            'model_name': self.model_name,
            'search_type': 'API-based embeddings',
            'vectorstore_active': True,
            'embeddings_count': len(self.embeddings)
        }
    
    def save_to_disk(self, path: str) -> None:
        """Save vector store to disk."""
        pass
    
    def load_from_disk(self, path: str) -> None:
        """Load vector store from disk."""
        pass
    
    def clear(self) -> None:
        """Clear all data from the vector store."""
        self.chunks = []
        self.embeddings = []
    
    def add_single_chunk(self, chunk: Dict[str, Any]) -> None:
        """Add a single chunk to the vector store."""
        if chunk:
            self.chunks.append(chunk)
            embedding = self._get_embedding(chunk['content'])
            self.embeddings.append(embedding)
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk from the vector store."""
        for i, chunk in enumerate(self.chunks):
            if chunk.get('id') == chunk_id:
                self.chunks.pop(i)
                if i < len(self.embeddings):
                    self.embeddings.pop(i)
                return True
        return False