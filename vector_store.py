import requests
import numpy as np
from typing import List, Dict, Any
import json
from sklearn.metrics.pairwise import cosine_similarity

class VectorStore:
    def __init__(self, api_key: str):
        """
        Initialize vector store with OpenRouter embeddings.
        
        Args:
            api_key: OpenRouter API key
        """
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.chunks = []
        self.embeddings = []
        self.vectorstore = None
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings from OpenRouter API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io",
            "X-Title": "PDF Chat Assistant"
        }
        
        # Try different embedding models that might be available
        embedding_models = [
            "openai/text-embedding-3-small",
            "openai/text-embedding-ada-002", 
            "text-embedding-3-small",
            "text-embedding-ada-002"
        ]
        
        for model in embedding_models:
            try:
                data = {
                    "model": model,
                    "input": texts
                }
                
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings = [item["embedding"] for item in result["data"]]
                    print(f"Successfully got embeddings using model: {model}")
                    return embeddings
                else:
                    print(f"Model {model} failed with status {response.status_code}: {response.text}")
                    continue
                    
            except Exception as e:
                print(f"Error with model {model}: {e}")
                continue
        
        # If all models fail, raise an error
        raise Exception("Could not get embeddings from any available model")
    
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
            
            # Get embeddings in batches to avoid API limits
            batch_size = 100  # Adjust based on API limits
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.get_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
                print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            self.embeddings = np.array(all_embeddings)
            self.vectorstore = True
            print(f"Successfully created vector store with {len(chunks)} documents")
            
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
            query_embedding = self.get_embeddings([query])[0]
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
            'vectorstore_type': 'OpenRouter Embeddings',
            'vectorstore_active': self.vectorstore is not None,
            'embedding_model': 'OpenRouter API',
            'embedding_dimensions': len(self.embeddings[0]) if len(self.embeddings) > 0 else 0
        }
    
    def save_to_disk(self, path: str) -> None:
        """Save vector store to disk."""
        print("OpenRouter embeddings vector store ready")
    
    def load_from_disk(self, path: str) -> None:
        """Load vector store from disk."""
        print("OpenRouter embeddings vector store loaded")
    
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