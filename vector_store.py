import math
import re
from typing import List, Dict, Any
from collections import Counter

class VectorStore:
    def __init__(self, model_name: str = "keyword_search"):
        """
        Initialize vector store with keyword-based search.
        
        Args:
            model_name: Model name (using keyword search for reliability)
        """
        self.model_name = model_name
        self.chunks = []
        self.vectorstore = True  # Always available
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for keyword search.
        
        Args:
            text: Input text
            
        Returns:
            List of words
        """
        # Convert to lowercase and extract words (including Russian)
        text = text.lower()
        words = re.findall(r'\b[a-zA-Zа-яё]{3,}\b', text)
        
        # Russian and English stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'has', 'let', 'put', 'say', 'she', 'too', 'use',
            'что', 'это', 'как', 'его', 'для', 'все', 'при', 'был', 'она', 'так', 'или', 'уже', 'раз', 'там', 'них', 'про', 'тем', 'где', 'этот', 'тоже', 'того', 'быть', 'если', 'есть', 'чтобы', 'более', 'после', 'можно', 'между'
        }
        
        return [word for word in words if word not in stop_words]
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add text chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
        """
        self.chunks = chunks
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using keyword matching and phrase search.
        
        Args:
            query: Search query text
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries with chunk content and similarity scores
        """
        if not self.chunks:
            return []
        
        query_words = self._preprocess_text(query)
        query_lower = query.lower()
        results = []
        
        for chunk in self.chunks:
            content_lower = chunk['content'].lower()
            
            # Exact phrase matching (highest priority)
            phrase_matches = 0
            query_phrases = [phrase.strip() for phrase in query.split() if len(phrase.strip()) > 2]
            for phrase in query_phrases:
                if phrase.lower() in content_lower:
                    phrase_matches += 1
            
            # Keyword matching
            chunk_words = self._preprocess_text(chunk['content'])
            keyword_matches = sum(1 for word in query_words if word in chunk_words)
            
            # Special scoring for Russian academic terms
            academic_terms = ['результат', 'достигнут', 'авторы', 'исследование', 'вывод', 'заключение', 'цель', 'задача', 'метод', 'анализ', 'данные', 'показ', 'выявл', 'установл', 'получ', 'опред']
            academic_matches = sum(1 for term in academic_terms if term in content_lower)
            
            # Calculate final score
            phrase_score = phrase_matches * 2.0
            keyword_score = keyword_matches / max(len(query_words), 1) * 1.0
            academic_score = academic_matches * 0.5
            
            final_score = phrase_score + keyword_score + academic_score
            
            # Boost score if chunk contains question-related terms
            question_boost = 0
            if any(word in content_lower for word in ['результат', 'итог', 'достиг', 'получ', 'выявл', 'показ', 'устан']):
                question_boost = 0.3
            
            final_score += question_boost
            
            if final_score > 0:
                result_chunk = chunk.copy()
                result_chunk['score'] = final_score
                result_chunk['phrase_matches'] = phrase_matches
                result_chunk['keyword_matches'] = keyword_matches
                result_chunk['academic_matches'] = academic_matches
                results.append(result_chunk)
        
        # Sort by score (descending) and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
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
            'search_type': 'Keyword + Phrase matching',
            'vectorstore_active': True
        }
    
    def save_to_disk(self, path: str) -> None:
        """Save vector store to disk (placeholder)."""
        pass
    
    def load_from_disk(self, path: str) -> None:
        """Load vector store from disk (placeholder)."""
        pass
    
    def clear(self) -> None:
        """Clear all data from the vector store."""
        self.chunks = []
    
    def add_single_chunk(self, chunk: Dict[str, Any]) -> None:
        """Add a single chunk to the vector store."""
        if chunk:
            self.chunks.append(chunk)
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk from the vector store."""
        original_length = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk.get('id') != chunk_id]
        return len(self.chunks) < original_length