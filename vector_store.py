from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Any
import tempfile
import os

class VectorStore:
    def __init__(self, api_key: str):
        """
        Initialize vector store with LangChain and FAISS.
        
        Args:
            api_key: OpenAI API key for embeddings
        """
        # Set up embeddings with API key
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )
        
        # Set up text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        self.vectorstore = None
        self.chunks = []
        self.documents = []
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add text chunks to the vector store using LangChain.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
        """
        if not chunks:
            return
        
        self.chunks = chunks
        
        # Convert chunks to LangChain Documents
        documents = []
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
        
        self.documents = documents
        
        # Create FAISS vector store from documents
        try:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            print(f"Successfully created FAISS vector store with {len(documents)} documents")
        except Exception as e:
            print(f"Error creating vector store: {e}")
            self.vectorstore = None
    
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
            print("Vector store not initialized")
            return []
        
        try:
            # Use LangChain's similarity search with scores
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Convert results to our format
            formatted_results = []
            for doc, distance in results:
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity_score = 1.0 / (1.0 + distance)
                
                if similarity_score >= score_threshold:
                    result = {
                        'content': doc.page_content,
                        'score': similarity_score,
                        'distance': distance,
                        'metadata': doc.metadata,
                        'id': doc.metadata.get('chunk_id', 'unknown')
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
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
            'vectorstore_type': 'FAISS with LangChain',
            'vectorstore_active': self.vectorstore is not None,
            'embedding_model': 'OpenAI via OpenRouter'
        }
    
    def save_to_disk(self, path: str) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Directory path to save the vector store
        """
        if self.vectorstore:
            try:
                self.vectorstore.save_local(path)
                print(f"Vector store saved to {path}")
            except Exception as e:
                print(f"Error saving vector store: {e}")
    
    def load_from_disk(self, path: str) -> None:
        """
        Load vector store from disk.
        
        Args:
            path: Directory path containing the saved vector store
        """
        try:
            self.vectorstore = FAISS.load_local(path, self.embeddings)
            print(f"Vector store loaded from {path}")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.clear()
    
    def clear(self) -> None:
        """
        Clear all data from the vector store.
        """
        self.chunks = []
        self.documents = []
        self.vectorstore = None
    
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