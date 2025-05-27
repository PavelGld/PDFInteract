from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Dict, Any
import tempfile
import os

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector store with LangChain and ChromaDB.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vectorstore = None
        self.chunks = []
        
        # Create temporary directory for ChromaDB
        self.temp_dir = tempfile.mkdtemp()
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add text chunks to the vector store using LangChain.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
        """
        if not chunks:
            return
        
        # Store original chunks for reference
        self.chunks = chunks
        
        # Convert chunks to LangChain Documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk['content'],
                metadata={
                    'chunk_id': chunk.get('id', f'chunk_{i}'),
                    'start_pos': chunk.get('start_pos', 0),
                    'end_pos': chunk.get('end_pos', 0)
                }
            )
            documents.append(doc)
        
        # Create ChromaDB vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.temp_dir
        )
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using LangChain similarity search.
        
        Args:
            query: Search query text
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries with chunk content and similarity scores
        """
        if not self.vectorstore:
            return []
        
        try:
            # Use LangChain's similarity search with scores
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Convert results to our format
            formatted_results = []
            for doc, score in results:
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity_score = 1.0 / (1.0 + score)
                
                if similarity_score >= score_threshold:
                    result = {
                        'content': doc.page_content,
                        'score': similarity_score,
                        'metadata': doc.metadata,
                        'id': doc.metadata.get('chunk_id', 'unknown')
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
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
            'vectorstore_type': 'ChromaDB with LangChain',
            'has_vectorstore': self.vectorstore is not None
        }
    
    def save_to_disk(self, path: str) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Directory path to save the vector store
        """
        if self.vectorstore:
            self.vectorstore.persist()
    
    def load_from_disk(self, path: str) -> None:
        """
        Load vector store from disk.
        
        Args:
            path: Directory path containing the saved vector store
        """
        try:
            self.vectorstore = Chroma(
                persist_directory=path,
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"Error loading from disk: {e}")
            self.clear()
    
    def clear(self) -> None:
        """
        Clear all data from the vector store.
        """
        self.chunks = []
        self.vectorstore = None
        
        # Create new temporary directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = tempfile.mkdtemp()
    
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
    
    def get_retriever(self, search_type: str = "similarity", k: int = 5):
        """
        Get LangChain retriever for the vector store.
        
        Args:
            search_type: Type of search to perform
            k: Number of documents to retrieve
            
        Returns:
            LangChain retriever object
        """
        if not self.vectorstore:
            return None
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )