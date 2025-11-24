"""
RAG Chain Module
Implements retrieval-augmented generation using LangChain v0.3.x
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from typing import Dict, List
import os


class RAGChain:
    """Handles RAG chain creation and query processing."""
    
    def __init__(
        self,
        vector_store: Chroma,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        k: int = 4
    ):
        """
        Initialize RAG chain.
        
        Args:
            vector_store: Chroma vector store instance
            model_name: OpenAI model to use
            temperature: Temperature for generation
            k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.k = k
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Get retriever
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Create prompt template (simple version without chat history in template)
        self.prompt = ChatPromptTemplate.from_template("""You are a helpful assistant that answers questions based on the provided context.

Use the following context to answer the user's question. If you cannot find the answer in the context, say so clearly.

Context:
{context}

Question: {question}

Guidelines:
- Only use information from the provided context
- Be precise and concise
- If the context doesn't contain relevant information, acknowledge it
- Cite specific parts of the context when possible

Answer:""")
    
    def format_docs(self, docs):
        """Format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def get_response(
        self,
        question: str,
        chat_history: List[Dict] = None
    ) -> Dict:
        """
        Get response from RAG chain.
        
        Args:
            question: User question
            chat_history: Previous conversation history
            
        Returns:
            Dictionary with answer and source documents
        """
        # Retrieve relevant documents
        source_documents = self.retriever.invoke(question)
        
        # Format context from retrieved documents
        context = self.format_docs(source_documents)
        
        # Create the chain with proper LangChain 0.3.x syntax
        # Using dictionary directly instead of RunnableParallel for compatibility
        chain = (
            {
                "context": lambda x: context,
                "question": lambda x: x
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Get response
        try:
            answer = chain.invoke(question)
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
            print(f"Chain error: {e}")
        
        return {
            "answer": answer,
            "source_documents": source_documents
        }
    
    def update_retriever(self, k: int):
        """
        Update number of documents to retrieve.
        
        Args:
            k: New number of documents
        """
        self.k = k
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )