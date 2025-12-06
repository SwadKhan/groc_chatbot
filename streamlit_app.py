"""
Dynamic Streamlit Web Interface for RAG Pipeline
Automatically loads vector store on startup
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import os
from basic_rag import BasicRAG
from advanced_rag import AdvancedRAG
import time

# Page config
st.set_page_config(
    page_title="RAG Pipeline Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stAlert {
        margin-top: 1rem;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore_loaded' not in st.session_state:
    st.session_state.vectorstore_loaded = False
if 'auto_load_attempted' not in st.session_state:
    st.session_state.auto_load_attempted = False
if 'documents_info' not in st.session_state:
    st.session_state.documents_info = []

def auto_load_vectorstore():
    """Automatically load vector store on startup"""
    if os.path.exists("./chroma_db"):
        try:
            with st.spinner("🔄 Loading knowledge base..."):
                rag = BasicRAG()
                rag.load_vectorstore("./chroma_db")
                rag.setup_chain(k=3, temperature=0)
                st.session_state.rag_system = rag
                st.session_state.vectorstore_loaded = True
                
                # Get document info
                if os.path.exists("documents"):
                    docs = [f for f in os.listdir("documents") if f.endswith('.pdf')]
                    st.session_state.documents_info = docs
                
                return True, f"✅ Loaded successfully! Ready to answer questions from {len(st.session_state.documents_info)} document(s)."
        except Exception as e:
            return False, f"❌ Error loading: {str(e)}"
    return False, "No existing knowledge base found."

def process_documents(files, doc_type: str, chunk_size: int, chunk_overlap: int):
    """Process uploaded documents"""
    try:
        temp_paths = []
        for file in files:
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            temp_paths.append(temp_path)
        
        with st.spinner("🔄 Processing documents... This may take a few minutes."):
            rag = BasicRAG(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            rag.load_documents(temp_paths, doc_type=doc_type)
            rag.create_vectorstore(persist_directory="./chroma_db")
            rag.setup_chain(k=3)
        
        # Clean up
        for path in temp_paths:
            if os.path.exists(path):
                os.remove(path)
        
        st.session_state.rag_system = rag
        st.session_state.vectorstore_loaded = True
        st.session_state.documents_info = [f.name for f in files]
        
        return True, f"✅ Successfully processed {len(files)} document(s)!"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def check_relevance(question: str, answer: str) -> bool:
    """Check if the answer indicates the question is out of scope"""
    out_of_scope_phrases = [
        "don't know",
        "not mentioned",
        "not provided",
        "not available in",
        "no information",
        "cannot find",
        "not found in the context",
        "based on the context provided, i don't",
        "the context does not"
    ]
    answer_lower = answer.lower()
    return not any(phrase in answer_lower for phrase in out_of_scope_phrases)

# Main app header
st.markdown('<p class="main-header">🤖 RAG Pipeline Demo</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Retrieval-Augmented Generation System</p>', unsafe_allow_html=True)

# Auto-load vector store on first run
if not st.session_state.auto_load_attempted:
    success, message = auto_load_vectorstore()
    st.session_state.auto_load_attempted = True
    if success:
        st.success(message)
    else:
        st.info("👋 Welcome! Please upload documents or load an existing knowledge base to get started.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # System Status
    if st.session_state.vectorstore_loaded:
        st.success("🟢 System Ready")
        if st.session_state.documents_info:
            with st.expander("📚 Loaded Documents", expanded=False):
                for doc in st.session_state.documents_info:
                    st.text(f"• {doc}")
    else:
        st.warning("🟡 No Knowledge Base Loaded")
    
    st.markdown("---")
    
    # Mode selection
    mode = st.radio(
        "Choose Action",
        ["💬 Chat with Documents", "📁 Manage Documents"],
        index=0 if st.session_state.vectorstore_loaded else 1
    )
    
    if mode == "📁 Manage Documents":
        st.markdown("---")
        
        # Quick load existing
        if st.button("🔄 Reload Knowledge Base", use_container_width=True):
            with st.spinner("Loading..."):
                success, message = auto_load_vectorstore()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        st.markdown("---")
        st.subheader("Upload New Documents")
        
        doc_type = st.selectbox("Document Type", ["pdf", "txt"])
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt']
        )
        
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("Chunk Size", value=1000, step=100, min_value=100)
        with col2:
            chunk_overlap = st.number_input("Chunk Overlap", value=200, step=50, min_value=0)
        
        if uploaded_files:
            if st.button("🚀 Process Documents", use_container_width=True):
                success, message = process_documents(uploaded_files, doc_type, chunk_size, chunk_overlap)
                if success:
                    st.success(message)
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(message)
    
    else:  # Chat mode
        st.markdown("---")
        st.subheader("Query Settings")
        
        use_advanced = st.checkbox("🚀 Advanced RAG", value=False)
        k_docs = st.slider("Documents to retrieve", 1, 10, 3)
        temperature = st.slider("Creativity", 0.0, 1.0, 0.0, 0.1)
        
        if use_advanced:
            strategy = st.selectbox(
                "Retrieval Strategy",
                ["basic", "multi_query", "hyde", "query_rewrite"]
            )
            use_rerank = st.checkbox("Use Reranking", value=True)
        
        if st.session_state.vectorstore_loaded:
            if st.button("Apply Settings", use_container_width=True):
                st.session_state.rag_system.setup_chain(k=k_docs, temperature=temperature)
                st.success("✅ Settings applied!")
        
        st.markdown("---")
        
        # Quick actions
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# Main content area
if not st.session_state.vectorstore_loaded:
    # Welcome screen
    st.info("👈 Please use the sidebar to upload documents or load an existing knowledge base.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📤 Upload Documents")
        st.write("Add your PDF or text files to create a new knowledge base")
    
    with col2:
        st.markdown("### 🔄 Load Existing")
        st.write("Load a previously created knowledge base")
    
    with col3:
        st.markdown("### 💬 Ask Questions")
        st.write("Chat with your documents in real-time")
    
    st.markdown("---")
    st.markdown("**Example Questions You Can Ask:**")
    st.code("""
• What is the main topic of the document?
• Summarize the key findings
• What methodology was used?
• Explain [specific concept] from the document
    """)

else:
    # Chat interface
    st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show relevance indicator
            if "relevant" in message:
                if message["relevant"]:
                    st.caption("✅ Answer based on your documents")
                else:
                    st.caption("⚠️ Question may be outside the scope of loaded documents")
            
            if "sources" in message:
                with st.expander(f"📚 View {len(message['sources'])} Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}**")
                        st.text(source["content"][:300] + "...")
                        st.caption(f"Metadata: {source['metadata']}")
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                start_time = time.time()
                
                try:
                    # Get response
                    if use_advanced and hasattr(st.session_state.rag_system, 'advanced_query'):
                        response = st.session_state.rag_system.advanced_query(
                            prompt,
                            strategy=strategy,
                            k=k_docs,
                            rerank=use_rerank
                        )
                    else:
                        response = st.session_state.rag_system.query(prompt, verbose=False)
                    
                    elapsed = time.time() - start_time
                    answer = response["result"]
                    
                    # Check relevance
                    is_relevant = check_relevance(prompt, answer)
                    
                    # Display answer with relevance check
                    if not is_relevant:
                        st.warning("⚠️ **Note:** This question may be outside the scope of the loaded documents.")
                        st.markdown(answer)
                        st.info("💡 **Tip:** Try asking questions specifically about the content in your uploaded documents.")
                    else:
                        st.markdown(answer)
                        st.caption(f"✅ Answer based on your documents")
                    
                    st.caption(f"⏱️ Response time: {elapsed:.2f}s")
                    
                    # Prepare sources
                    sources = [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in response["source_documents"]
                    ]
                    
                    # Add to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "relevant": is_relevant
                    })
                    
                    # Show sources
                    with st.expander(f"📚 View {len(sources)} Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}**")
                            st.text(source["content"][:300] + "...")
                            st.caption(f"Metadata: {source['metadata']}")
                            st.markdown("---")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Please try rephrasing your question or check if the documents are loaded correctly.")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🔧 Built with LangChain & OpenAI")
with col2:
    st.caption("📊 Real-time Document Analysis")
with col3:
    st.caption("🚀 Powered by RAG Technology")