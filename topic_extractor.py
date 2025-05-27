"""
Topic Extractor для PDF Chat Assistant

Модуль для извлечения тематических меток из PDF документов с учетом изображений.
Использует LLM для анализа контента и генерации релевантных тем.
"""

import re
from typing import List, Dict, Any, Optional
from collections import Counter
import base64


class TopicExtractor:
    def __init__(self, openrouter_client=None):
        """
        Initialize topic extractor.
        
        Args:
            openrouter_client: OpenRouter client for LLM-based topic extraction
        """
        self.openrouter_client = openrouter_client
        
    def extract_keywords_basic(self, text: str, max_keywords: int = 20) -> List[str]:
        """
        Extract basic keywords using frequency analysis.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords
        """
        # Clean text and extract words
        words = re.findall(r'\b[а-яё]+\b|\b[a-z]+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'при', 'или', 'что', 'как', 'из',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'
        }
        
        # Filter words by length and stop words
        filtered_words = [
            word for word in words 
            if len(word) > 3 and word not in stop_words
        ]
        
        # Count frequency and return top keywords
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    def extract_topics_llm(self, text: str, images_description: str = "", model: str = "openai/gpt-4o-mini") -> List[str]:
        """
        Extract topics using LLM analysis.
        
        Args:
            text: Document text
            images_description: Description of images in the document
            model: LLM model to use
            
        Returns:
            List of topic tags
        """
        if not self.openrouter_client:
            return self.extract_keywords_basic(text, 10)
            
        try:
            # Create prompt for topic extraction
            prompt = f"""
            Проанализируй следующий документ и извлеки 8-12 основных тематических меток на русском языке.
            Метки должны быть краткими (1-3 слова) и отражать ключевые темы документа.
            
            Текст документа:
            {text[:3000]}...
            
            {"Описание изображений в документе: " + images_description if images_description else ""}
            
            Верни только список тем через запятую, без нумерации и дополнительных пояснений.
            Пример: технологии, образование, бизнес, инновации
            """
            
            response = self.openrouter_client.get_response(
                messages=[],
                question=prompt,
                context="",
                model=model,
                max_tokens=200,
                temperature=0.3
            )
            
            # Parse response to extract topics
            topics = [topic.strip() for topic in response.split(',')]
            topics = [topic for topic in topics if len(topic) > 0 and len(topic) < 30]
            
            return topics[:12]  # Limit to 12 topics
            
        except Exception as e:
            print(f"Error extracting topics with LLM: {e}")
            return self.extract_keywords_basic(text, 10)
    
    def analyze_document_structure(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze document structure and extract metadata.
        
        Args:
            chunks: Document chunks
            
        Returns:
            Document structure analysis
        """
        total_text = " ".join([chunk.get('content', '') for chunk in chunks])
        
        analysis = {
            'total_chunks': len(chunks),
            'total_words': len(total_text.split()),
            'total_chars': len(total_text),
            'avg_chunk_size': len(total_text) // len(chunks) if chunks else 0,
            'has_technical_content': self._detect_technical_content(total_text),
            'language': self._detect_language(total_text),
            'document_type': self._classify_document_type(total_text)
        }
        
        return analysis
    
    def _detect_technical_content(self, text: str) -> bool:
        """Check if document contains technical content."""
        technical_indicators = [
            'алгоритм', 'технология', 'система', 'метод', 'процесс', 'данные',
            'algorithm', 'technology', 'system', 'method', 'process', 'data'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in technical_indicators)
    
    def _detect_language(self, text: str) -> str:
        """Detect primary language of the document."""
        # Simple heuristic based on character frequency
        cyrillic_chars = len(re.findall(r'[а-яё]', text.lower()))
        latin_chars = len(re.findall(r'[a-z]', text.lower()))
        
        if cyrillic_chars > latin_chars:
            return 'russian'
        else:
            return 'english'
    
    def _classify_document_type(self, text: str) -> str:
        """Classify document type based on content."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['статья', 'исследование', 'article', 'research']):
            return 'academic'
        elif any(word in text_lower for word in ['отчет', 'доклад', 'report', 'presentation']):
            return 'business'
        elif any(word in text_lower for word in ['инструкция', 'руководство', 'manual', 'guide']):
            return 'manual'
        else:
            return 'general'
    
    def generate_document_summary(self, text: str, topics: List[str]) -> str:
        """
        Generate a brief document summary.
        
        Args:
            text: Document text
            topics: Extracted topics
            
        Returns:
            Document summary
        """
        if not self.openrouter_client:
            return f"Документ содержит {len(text.split())} слов. Основные темы: {', '.join(topics[:5])}"
            
        try:
            prompt = f"""
            Создай краткое описание документа (2-3 предложения) на основе его содержания и тем.
            
            Основные темы: {', '.join(topics)}
            
            Начало документа:
            {text[:1000]}...
            
            Описание должно быть информативным и кратким.
            """
            
            response = self.openrouter_client.get_response(
                messages=[],
                question=prompt,
                context="",
                model="openai/gpt-4o-mini",
                max_tokens=150,
                temperature=0.5
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Документ содержит {len(text.split())} слов. Основные темы: {', '.join(topics[:5])}"