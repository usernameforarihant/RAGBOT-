"""
Vector Store Module
Manages Chroma DB with persistent storage using LangChain v0.3.x
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Optional
from langchain_core.documents import Document
import os


class VectorStoreManager:
    """Manages Chroma vector store with persistence."""
    
    def __init__(self, persist_directory: str):
        """
        Initialize vector store manager.
        
        Args:
            persist_directory: Directory for persistent storage
        """
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.vector_store: Optional[Chroma] = None
        self.collection_name: Optional[str] = None
    
    def create_vector_store(
        self,
        documents: List[Document],
        collection_name: str
    ) -> Chroma:
        """
        Create new vector store from documents.
        
        Args:
            documents: List of documents to embed
            collection_name: Name for the collection
            
        Returns:
            Chroma vector store instance
        """
        self.collection_name = collection_name
        
        # Create collection-specific directory
        collection_path = os.path.join(self.persist_directory, collection_name)
        os.makedirs(collection_path, exist_ok=True)
        
        # Create vector store with persistence
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=collection_path
        )
        
        return self.vector_store
    
    def load_existing(self, collection_name: str) -> Chroma:
        """
        Load existing vector store from disk.
        
        Args:
            collection_name: Name of the collection to load
            
        Returns:
            Chroma vector store instance
        """
        self.collection_name = collection_name
        collection_path = os.path.join(self.persist_directory, collection_name)
        
        # Load existing vector store
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=collection_path
        )
        
        return self.vector_store
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to existing vector store.
        
        Args:
            documents: List of documents to add
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        self.vector_store.add_documents(documents)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def get_retriever(self, k: int = 4):
        """
        Get retriever for RAG chain.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            Retriever instance
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k