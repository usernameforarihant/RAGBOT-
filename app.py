"""
RAG Chatbot with Streamlit UI
LangChain v0.3.x compatible implementation
"""

import streamlit as st
import os
from pathlib import Path
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStoreManager
from utils.rag_chain import RAGChain
from utils.memory import ConversationMemory
from dotenv import load_dotenv
load_dotenv()

# Configuration
UPLOAD_DIR = "data/uploaded_pdfs"
CHROMA_DIR = "data/chroma_db"
CONVERSATION_DIR = "conversations"

# Create necessary directories
for dir_path in [UPLOAD_DIR, CHROMA_DIR, CONVERSATION_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "memory" not in st.session_state:
        st.session_state.memory = None


def get_uploaded_pdfs():
    """Get list of already uploaded PDFs."""
    if not os.path.exists(UPLOAD_DIR):
        return []
    return [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.pdf')]


def check_embeddings_exist(pdf_name):
    """Check if embeddings already exist for a PDF."""
    collection_name = pdf_name.replace('.pdf', '').replace(' ', '_').lower()
    vector_store_path = os.path.join(CHROMA_DIR, collection_name)
    return os.path.exists(vector_store_path) and os.listdir(vector_store_path)


def process_pdf(pdf_file, pdf_name):
    """Process uploaded PDF and create/load embeddings."""
    with st.spinner(f"Processing {pdf_name}..."):
        # Save PDF to disk
        pdf_path = os.path.join(UPLOAD_DIR, pdf_name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        # Check if embeddings exist
        collection_name = pdf_name.replace('.pdf', '').replace(' ', '_').lower()
        
        if check_embeddings_exist(pdf_name):
            st.info(f"✓ Found existing embeddings for {pdf_name}. Loading...")
            # Load existing vector store
            vector_store_manager = VectorStoreManager(CHROMA_DIR)
            vector_store_manager.load_existing(collection_name)
        else:
            st.info(f"Creating new embeddings for {pdf_name}...")
            # Process PDF and create embeddings
            pdf_processor = PDFProcessor()
            documents = pdf_processor.load_pdf(pdf_path)
            text_chunks = pdf_processor.split_documents(documents)
            
            # Create vector store
            vector_store_manager = VectorStoreManager(CHROMA_DIR)
            vector_store_manager.create_vector_store(text_chunks, collection_name)
            st.success(f"✓ Created {len(text_chunks)} embeddings for {pdf_name}")
        
        return vector_store_manager, collection_name


def load_pdf_for_chat(pdf_name):
    """Load selected PDF for chatting."""
    if not pdf_name:
        return
    
    # Reset chat history when switching PDFs
    if st.session_state.current_pdf != pdf_name:
        st.session_state.messages = []
        st.session_state.current_pdf = pdf_name
    
    # Load or create vector store
    collection_name = pdf_name.replace('.pdf', '').replace(' ', '_').lower()
    pdf_path = os.path.join(UPLOAD_DIR, pdf_name)
    
    if check_embeddings_exist(pdf_name):
        # Load existing embeddings
        vector_store_manager = VectorStoreManager(CHROMA_DIR)
        vector_store_manager.load_existing(collection_name)
    else:
        # Process PDF if embeddings don't exist
        pdf_processor = PDFProcessor()
        documents = pdf_processor.load_pdf(pdf_path)
        text_chunks = pdf_processor.split_documents(documents)
        
        vector_store_manager = VectorStoreManager(CHROMA_DIR)
        vector_store_manager.create_vector_store(text_chunks, collection_name)
    
    # Initialize RAG chain and memory
    st.session_state.vector_store_manager = vector_store_manager
    st.session_state.rag_chain = RAGChain(vector_store_manager.vector_store)
    
    # Load conversation memory for this PDF
    memory_file = os.path.join(CONVERSATION_DIR, f"{collection_name}_memory.json")
    st.session_state.memory = ConversationMemory(memory_file)
    st.session_state.messages = st.session_state.memory.load_history()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 RAG Chatbot with PDF Knowledge Base")
    st.markdown("Upload PDFs and chat with your documents using AI-powered retrieval")
    
    # Initialize session state
    initialize_session_state()
    
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        st.error("⚠️ OPENAI_API_KEY environment variable not set!")
        st.info("Please set your OpenAI API key: `export OPENAI_API_KEY='your-key-here'`")
        st.stop()
    
    # Sidebar for PDF management
    with st.sidebar:
        st.header("📄 Document Management")
        
        # Upload new PDF
        st.subheader("Upload New PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF to add to your knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("Process PDF", type="primary"):
                vector_store_manager, collection_name = process_pdf(
                    uploaded_file,
                    uploaded_file.name
                )
                st.success(f"✓ {uploaded_file.name} processed successfully!")
                st.rerun()
        
        st.divider()
        
        # Select existing PDF
        st.subheader("Select Document for Chat")
        existing_pdfs = get_uploaded_pdfs()
        
        if existing_pdfs:
            selected_pdf = st.selectbox(
                "Available PDFs",
                options=[""] + existing_pdfs,
                format_func=lambda x: "Select a PDF..." if x == "" else x
            )
            
            if selected_pdf and selected_pdf != st.session_state.current_pdf:
                if st.button("Load PDF", type="primary"):
                    load_pdf_for_chat(selected_pdf)
                    st.success(f"✓ Loaded {selected_pdf}")
                    st.rerun()
            
            # Show current PDF
            if st.session_state.current_pdf:
                st.info(f"📖 Current: {st.session_state.current_pdf}")
                
                # Show embedding status
                if check_embeddings_exist(st.session_state.current_pdf):
                    st.success("✓ Embeddings loaded")
        else:
            st.warning("No PDFs uploaded yet. Upload a PDF to get started!")
        
        st.divider()
        
        # Clear chat history button
        if st.session_state.current_pdf:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                if st.session_state.memory:
                    st.session_state.memory.clear_history()
                st.rerun()
    
    # Main chat interface
    if not st.session_state.current_pdf:
        st.info("👈 Please select or upload a PDF from the sidebar to start chatting")
        st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source[:300] + "..." if len(source) > 300 else source)
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from RAG chain
                    response = st.session_state.rag_chain.get_response(
                        prompt,
                        st.session_state.messages
                    )
                    
                    # Display response
                    st.markdown(response["answer"])
                    
                    # Display sources
                    if response.get("source_documents"):
                        with st.expander("📚 View Sources"):
                            for i, doc in enumerate(response["source_documents"], 1):
                                st.markdown(f"**Source {i}:**")
                                st.text(doc.page_content[:300] + "..." 
                                       if len(doc.page_content) > 300 
                                       else doc.page_content)
                                st.divider()
                    
                    # Add assistant message to chat
                    sources = [doc.page_content for doc in response.get("source_documents", [])]
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": sources
                    })
                    
                    # Save conversation history
                    if st.session_state.memory:
                        st.session_state.memory.save_history(st.session_state.messages)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.error("Please try again or check your OpenAI API key.")


if __name__ == "__main__":
    main()