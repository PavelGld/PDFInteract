import math
import re
from typing import List, Dict, Any
from collections import Counter

class VectorStore:
    def __init__(self, model_name: str = "simple_tfidf"):
        """
        Initialize vector store with simple TF-IDF vectorization.
        
        Args:
            model_name: Model name (using simple TF-IDF for compatibility)
        """
        self.model_name = model_name
        self.chunks = []
        self.vocabulary = set()
        self.word_to_index = {}
        self.tfidf_vectors = []
        self.idf_scores = {}
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for TF-IDF calculation.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed words
        """
        # Convert to lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        
        # Simple stop word removal
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        return [word for word in words if word not in stop_words]
    
    def _build_tfidf_vectors(self):
        """
        Build TF-IDF vectors for all chunks.
        """
        if not self.chunks:
            return
        
        # Extract all words to build vocabulary
        all_words = set()
        processed_chunks = []
        
        for chunk in self.chunks:
            words = self._preprocess_text(chunk['content'])
            processed_chunks.append(words)
            all_words.update(words)
        
        self.vocabulary = sorted(list(all_words))
        self.word_to_index = {word: i for i, word in enumerate(self.vocabulary)}
        
        # Calculate IDF scores
        num_docs = len(processed_chunks)
        for word in self.vocabulary:
            doc_count = sum(1 for words in processed_chunks if word in words)
            self.idf_scores[word] = math.log(num_docs / (doc_count + 1))
        
        # Build TF-IDF vectors
        self.tfidf_vectors = []
        for words in processed_chunks:
            vector = self._create_tfidf_vector(words)
            self.tfidf_vectors.append(vector)
    
    def _create_tfidf_vector(self, words: List[str]) -> List[float]:
        """
        Create TF-IDF vector for a list of words.
        
        Args:
            words: List of preprocessed words
            
        Returns:
            TF-IDF vector
        """
        word_counts = Counter(words)
        total_words = len(words)
        
        vector = [0.0] * len(self.vocabulary)
        
        for word, count in word_counts.items():
            if word in self.word_to_index:
                tf = count / total_words
                idf = self.idf_scores.get(word, 0)
                tfidf = tf * idf
                vector[self.word_to_index[word]] = tfidf
        
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
        if not vec1 or not vec2:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add text chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
        """
        if not chunks:
            return
        
        # Store chunks
        self.chunks.extend(chunks)
        
        # Build vocabulary and TF-IDF vectors
        self._build_tfidf_vectors()
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using TF-IDF cosine similarity and keyword matching.
        
        Args:
            query: Search query text
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries with chunk content and similarity scores
        """
        if not self.chunks or not self.tfidf_vectors:
            return []
        
        # Create query vector
        query_words = self._preprocess_text(query)
        query_vector = self._create_tfidf_vector(query_words)
        
        # Calculate similarities with both TF-IDF and keyword matching
        similarities = []
        query_lower = query.lower()
        
        for i, chunk_vector in enumerate(self.tfidf_vectors):
            # TF-IDF cosine similarity
            tfidf_score = self._cosine_similarity(query_vector, chunk_vector)
            
            # Keyword matching boost
            chunk_content_lower = self.chunks[i]['content'].lower()
            keyword_matches = 0
            for word in query_words:
                if word in chunk_content_lower:
                    keyword_matches += 1
            
            # Combine scores with keyword boost
            keyword_boost = keyword_matches / max(len(query_words), 1) * 0.3
            final_score = tfidf_score + keyword_boost
            
            # Also check for exact phrase matches
            if len(query.strip()) > 3 and query.strip().lower() in chunk_content_lower:
                final_score += 0.5
            
            if final_score >= score_threshold:
                chunk = self.chunks[i].copy()
                chunk['score'] = final_score
                chunk['tfidf_score'] = tfidf_score
                chunk['keyword_matches'] = keyword_matches
                similarities.append(chunk)
        
        # If no good matches found, try with lower threshold
        if not similarities and score_threshold > 0.01:
            return self.search(query, k, 0.01)
        
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
            'vocabulary_size': len(self.vocabulary),
            'vector_dimension': len(self.vocabulary)
        }
    
    def save_to_disk(self, path: str) -> None:
        """
        Save vector store to disk (simplified implementation).
        
        Args:
            path: Directory path to save the vector store
        """
        # For simplicity, we'll just store basic info
        # In a real implementation, you'd save the full state
        pass
    
    def load_from_disk(self, path: str) -> None:
        """
        Load vector store from disk (simplified implementation).
        
        Args:
            path: Directory path containing the saved vector store
        """
        # For simplicity, we'll just clear the current state
        # In a real implementation, you'd load the full state
        self.clear()
    
    def clear(self) -> None:
        """
        Clear all data from the vector store.
        """
        self.chunks = []
        self.vocabulary = set()
        self.word_to_index = {}
        self.tfidf_vectors = []
        self.idf_scores = {}
    
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
        
        # Rebuild vectors
        if self.chunks:
            self._build_tfidf_vectors()
        else:
            self.clear()
        
        return True