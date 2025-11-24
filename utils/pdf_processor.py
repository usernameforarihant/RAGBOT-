"""
PDF Processing Module
Handles PDF loading and text splitting using LangChain v0.3.x
"""

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document


class PDFProcessor:
    """Handles PDF loading and document processing."""
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 300):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load PDF using PyMuPDFLoader.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata['source'] = pdf_path
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split document chunks
        """
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Complete pipeline: load and split PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of processed document chunks
        """
        documents = self.load_pdf(pdf_path)
        chunks = self.split_documents(documents)
        return chunks