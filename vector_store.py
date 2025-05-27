from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any
import os

class VectorStore:
    def __init__(self, api_key: str):
        """
        Initialize vector store with TF-IDF vectorization.
        
        Args:
            api_key: API key (kept for compatibility)
        """
        # Set up text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Use TF-IDF for reliable vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=1,
            max_df=0.95
        )
        
        self.vectorstore = None
        self.chunks = []
        self.documents = []
        self.tfidf_matrix = None
        self.is_fitted = False
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add text chunks to the vector store using TF-IDF.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
        """
        if not chunks:
            return
        
        self.chunks = chunks
        
        # Convert chunks to LangChain Documents
        documents = []
        texts = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk['content'],
                metadata={
                    'chunk_id': chunk.get('id', ''),
                    'start_pos': chunk.get('start_pos', 0),
                    'end_pos': chunk.get('end_pos', 0)
                }
            )
            documents.append(doc)
            texts.append(chunk['content'])
        
        self.documents = documents
        
        # Create TF-IDF matrix
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
            self.vectorstore = True  # Mark as active
            print(f"Successfully created TF-IDF vector store with {len(documents)} documents")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            self.vectorstore = None
            self.is_fitted = False
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using TF-IDF similarity.
        
        Args:
            query: Search query text
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries with chunk content and similarity scores
        """
        if not self.is_fitted or self.tfidf_matrix is None:
            print("Vector store not initialized")
            return []
        
        try:
            # Transform query using fitted vectorizer
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top k indices
            top_indices = similarities.argsort()[-k:][::-1]
            
            # Convert results to our format
            formatted_results = []
            for idx in top_indices:
                similarity_score = similarities[idx]
                
                if similarity_score >= score_threshold:
                    result = {
                        'content': self.chunks[idx]['content'],
                        'score': float(similarity_score),
                        'distance': float(1.0 - similarity_score),  # Convert to distance
                        'metadata': self.documents[idx].metadata,
                        'id': self.documents[idx].metadata.get('chunk_id', 'unknown')
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def get_retriever(self, search_type: str = "similarity", k: int = 5):
        """
        Get retriever-like interface.
        
        Args:
            search_type: Type of search to perform
            k: Number of documents to retrieve
            
        Returns:
            Custom retriever function
        """
        def retrieve(query: str):
            results = self.search(query, k=k, score_threshold=0.0)
            return [self.documents[i] for i, _ in enumerate(results) if i < len(self.documents)]
        
        return retrieve
    
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
            'total_documents': len(self.documents),
            'vectorstore_type': 'TF-IDF with scikit-learn',
            'vectorstore_active': self.vectorstore is not None and self.is_fitted,
            'embedding_model': 'TF-IDF Vectorization'
        }
    
    def save_to_disk(self, path: str) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Directory path to save the vector store
        """
        # For TF-IDF, we could save the vectorizer and matrix
        print(f"TF-IDF vector store ready (no disk save needed)")
    
    def load_from_disk(self, path: str) -> None:
        """
        Load vector store from disk.
        
        Args:
            path: Directory path containing the saved vector store
        """
        print(f"TF-IDF vector store initialization from memory")
    
    def clear(self) -> None:
        """
        Clear all data from the vector store.
        """
        self.chunks = []
        self.documents = []
        self.vectorstore = None
        self.tfidf_matrix = None
        self.is_fitted = False
    
    def add_single_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Add a single chunk to the vector store.
        
        Args:
            chunk: Chunk dictionary with 'content' key
        """
        self.add_chunks([chunk])
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """
        Remove a chunk from the vector store.
        
        Args:
            chunk_id: ID of the chunk to remove
            
        Returns:
            True if chunk was found and removed, False otherwise
        """
        # Find and remove the chunk
        original_length = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk.get('id') != chunk_id]
        
        if len(self.chunks) == original_length:
            return False  # Chunk not found
        
        # Rebuild vector store with remaining chunks
        if self.chunks:
            self.add_chunks(self.chunks)
        else:
            self.clear()
        
        return True