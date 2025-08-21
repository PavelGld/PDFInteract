"""
PDF Processor для PDF Chat Assistant

Модуль для извлечения и обработки текста из PDF документов.
Разбивает документы на фрагменты для эффективного векторного поиска.

Основные функции:
- Извлечение текста из PDF файлов
- Очистка и нормализация текста
- Разбиение на фрагменты с перекрытием
- Статистика документа
"""

import PyPDF2
import re
from typing import List, Dict, Any, Optional
from io import BytesIO
import pymupdf as fitz  # PyMuPDF
from PIL import Image
import base64
import io

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor with chunking parameters.
        
        Args:
            chunk_size: Maximum size of each text chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content as a string
        """
        try:
            text_content = ""
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # Clean up the text
                            page_text = self._clean_text(page_text)
                            text_content += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
                    except Exception as e:
                        print(f"Error extracting text from page {page_num + 1}: {e}")
                        continue
            
            return text_content.strip()
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove page numbers and headers/footers (simple heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that might be page numbers or artifacts
            if len(line) < 3:
                continue
            # Skip lines that are just numbers (likely page numbers)
            if line.isdigit():
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def create_chunks(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks for better context preservation.
        
        Args:
            text: Text content to chunk
            
        Returns:
            List of dictionaries with chunk content and metadata
        """
        if not text:
            return []
        
        chunks = []
        text_length = len(text)
        start = 0
        chunk_id = 0
        
        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size
            
            # If this isn't the last chunk, try to break at a sentence or paragraph
            if end < text_length:
                # Look for sentence endings within the overlap region
                search_start = max(start + self.chunk_size - self.chunk_overlap, start)
                search_end = min(end + self.chunk_overlap, text_length)
                
                # Find the best break point (sentence ending)
                sentence_breaks = []
                for i in range(search_start, search_end):
                    if text[i] in '.!?\n':
                        sentence_breaks.append(i)
                
                if sentence_breaks:
                    # Use the sentence break closest to our target end
                    end = min(sentence_breaks, key=lambda x: abs(x - end)) + 1
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'id': f"chunk_{chunk_id}",
                    'content': chunk_text,
                    'start_pos': start,
                    'end_pos': end,
                    'length': len(chunk_text)
                })
                chunk_id += 1
            
            # Move start position (with overlap)
            start = end - self.chunk_overlap
            
            # Ensure we make progress
            if start <= chunks[-1]['start_pos'] if chunks else False:
                start = end
        
        return chunks
    
    def get_chunk_context(self, chunks: List[Dict[str, str]], chunk_id: str, context_size: int = 2) -> str:
        """
        Get surrounding context for a specific chunk.
        
        Args:
            chunks: List of all chunks
            chunk_id: ID of the target chunk
            context_size: Number of chunks before and after to include
            
        Returns:
            Combined text with context
        """
        try:
            # Find the target chunk index
            target_idx = next(i for i, chunk in enumerate(chunks) if chunk['id'] == chunk_id)
            
            # Calculate context window
            start_idx = max(0, target_idx - context_size)
            end_idx = min(len(chunks), target_idx + context_size + 1)
            
            # Combine chunks in the context window
            context_chunks = chunks[start_idx:end_idx]
            context_text = '\n\n'.join([chunk['content'] for chunk in context_chunks])
            
            return context_text
            
        except (StopIteration, IndexError):
            # If chunk not found, return empty string
            return ""
    
    def get_document_stats(self, text: str) -> Dict[str, int]:
        """
        Get basic statistics about the document.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with document statistics
        """
        if not text:
            return {
                'total_characters': 0,
                'total_words': 0,
                'total_lines': 0,
                'total_paragraphs': 0
            }
        
        words = text.split()
        lines = text.split('\n')
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        return {
            'total_characters': len(text),
            'total_words': len(words),
            'total_lines': len(lines),
            'total_paragraphs': len(paragraphs)
        }
    
    def extract_images(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract images from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of image dictionaries with metadata
        """
        images = []
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Convert to RGB if necessary
                        if pix.n - pix.alpha < 4:
                            img_data = pix.tobytes("png")
                        else:
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None
                        
                        # Encode to base64 for storage
                        img_base64 = base64.b64encode(img_data).decode()
                        
                        images.append({
                            'page': page_num + 1,
                            'index': img_index,
                            'width': pix.width,
                            'height': pix.height,
                            'data': img_base64,
                            'size_bytes': len(img_data)
                        })
                        
                        pix = None  # Free memory
                        
                    except Exception as e:
                        print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            print(f"Error extracting images: {e}")
        
        return images
    
    def analyze_images_with_llm(self, images: List[Dict[str, Any]], openrouter_client=None, model: str = "openai/gpt-4o") -> str:
        """
        Analyze images using vision-capable LLM.
        
        Args:
            images: List of image dictionaries
            openrouter_client: OpenRouter client
            model: Vision-capable model
            
        Returns:
            Description of images content
        """
        if not images or not openrouter_client:
            return ""
        
        try:
            # Basic image description for now
            descriptions = []
            for img in images[:5]:  # Limit to first 5 images
                descriptions.append(f"Изображение на странице {img['page']} размером {img['width']}x{img['height']} пикселей")
            
            return "; ".join(descriptions)
            
        except Exception as e:
            print(f"Error analyzing images: {e}")
            return ""
