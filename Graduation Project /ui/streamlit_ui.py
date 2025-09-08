import streamlit as st
import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import RAG components (same as main notebook)
try:
    import chromadb
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.llms import HuggingFacePipeline
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="📚 Speak with Your Book",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.user-message {
    background-color: #e3f2fd;
    margin-left: 20%;
}
.bot-message {
    background-color: #f5f5f5;
    margin-right: 20%;
}
.source-info {
    font-size: 0.8rem;
    color: #666;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Configuration class
@st.cache_resource
class RAGConfig:
    MODEL_PATH = "development/outputs/fine_tuned_model"
    DATABASE_PATH = "development/outputs/vector_database"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RETRIEVAL_K = 5

config = RAGConfig()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system_loaded" not in st.session_state:
    st.session_state.rag_system_loaded = False
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

@st.cache_resource
def load_rag_system():
    """Load the RAG system components"""
    try:
        with st.spinner("🤖 Loading fine-tuned model..."):
            # Load model and tokenizer
            if os.path.exists(config.MODEL_PATH):
                tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
                model = AutoModelForCausalLM.from_pretrained(
                    config.MODEL_PATH,
                    torch_dtype=torch.float16 if config.DEVICE == "cuda" else torch.float32,
                    device_map="auto" if config.DEVICE == "cuda" else None,
                    trust_remote_code=True
                )
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True,
                    device=0 if config.DEVICE == "cuda" else -1
                )
                llm = HuggingFacePipeline(pipeline=pipe)
            else:
                st.warning("Fine-tuned model not found, using default model")
                llm = HuggingFacePipeline.from_model_id(
                    model_id="Qwen/Qwen-1_8B-Chat",
                    task="text-generation",
                    model_kwargs={"temperature": 0.7, "max_length": 512}
                )
        
        with st.spinner("🗄️ Loading vector database..."):
            # Load embeddings and database
            embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': config.DEVICE}
            )
            
            if os.path.exists(config.DATABASE_PATH):
                chroma_client = chromadb.PersistentClient(path=config.DATABASE_PATH)
                vectorstore = Chroma(
                    client=chroma_client,
                    embedding_function=embeddings,
                    collection_name="book_collection"
                )
                
                # Test database
                doc_count = vectorstore._collection.count()
                st.success(f"📚 Database loaded: {doc_count} documents")
            else:
                st.error("Vector database not found!")
                return None, None, 0
        
        with st.spinner("🔗 Setting up RAG chain..."):
            # Create RAG chain
            prompt_template = """
            You are an intelligent book assistant. Use the following context from the book to answer the question accurately and comprehensively.

            Context from book:
            {context}

            Question: {question}

            Instructions:
            - Answer based on the provided context
            - If the answer is not in the context, say "I cannot find this information in the book"
            - Be specific and cite relevant details
            - Keep your answer focused and relevant

            Answer:"""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
        
        return rag_chain, vectorstore, doc_count
    
    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        return None, None, 0

# Main app header
st.markdown('<h1 class="main-header">📚 Speak with Your Book</h1>', unsafe_allow_html=True)
st.markdown("### 🤖 AI-Powered Book Conversations using RAG")

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    
    # Load RAG system
    if not st.session_state.rag_system_loaded:
        if st.button("🚀 Initialize RAG System", type="primary"):
            rag_chain, vectorstore, doc_count = load_rag_system()
            if rag_chain:
                st.session_state.rag_chain = rag_chain
                st.session_state.vectorstore = vectorstore
                st.session_state.doc_count = doc_count
                st.session_state.rag_system_loaded = True
                st.success("✅ System loaded successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to load system")
    else:
        st.success("✅ RAG System Ready")
        st.info(f"📚 Documents: {st.session_state.doc_count}")
        st.info(f"🖥️ Device: {config.DEVICE}")
        
        if st.button("🔄 Reload System"):
            st.session_state.rag_system_loaded = False
            st.session_state.rag_chain = None
            st.rerun()
    
    st.divider()
    
    # System info
    st.subheader("⚙️ Configuration")
    st.text(f"Model: Fine-tuned Qwen")
    st.text(f"Retrieval: {config.RETRIEVAL_K} chunks")
    st.text(f"Embedding: MiniLM-L6")
    
    st.divider()
    
    # Chat controls
    st.subheader("💬 Chat Controls")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Example questions
    st.subheader("💡 Example Questions")
    example_questions = [
        "What is the main theme?",
        "Who are the main characters?", 
        "Summarize chapter 1",
        "What's the conclusion?",
        "Author's main argument?"
    ]
    
    for question in example_questions:
        if st.button(f"❓ {question}", key=f"example_{question}"):
            if st.session_state.rag_system_loaded:
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

# Main chat interface
if not st.session_state.rag_system_loaded:
    st.warning("⚠️ Please initialize the RAG system using the sidebar button")
    st.info("📋 Make sure you have:")
    st.markdown("""
    - ✅ Fine-tuned model in `development/outputs/fine_tuned_model/`
    - ✅ Vector database in `development/outputs/vector_database/`
    - ✅ All dependencies installed (`pip install -r requirements.txt`)
    """)
else:
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">🤔 <b>You:</b> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">🤖 <b>Assistant:</b> {message["content"]}</div>', unsafe_allow_html=True)
            if "sources" in message:
                st.markdown(f'<div class="source-info">📚 Sources used: {message["sources"]}</div>', unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask me anything about your book..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        st.markdown(f'<div class="chat-message user-message">🤔 <b>You:</b> {prompt}</div>', unsafe_allow_html=True)
        
        # Get AI response
        with st.spinner("🤖 Thinking..."):
            try:
                response = st.session_state.rag_chain({"query": prompt})
                answer = response["result"]
                sources = response["source_documents"]
                
                # Display AI response
                st.markdown(f'<div class="chat-message bot-message">🤖 <b>Assistant:</b> {answer}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="source-info">📚 Sources used: {len(sources)} chunks</div>', unsafe_allow_html=True)
                
                # Add to messages
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": len(sources)
                })
                
                # Show source details in expander
                with st.expander(f"🔍 View Source Chunks ({len(sources)} found)"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.markdown(f"```\n{source.page_content[:300]}...\n```")
                        if hasattr(source, 'metadata') and source.metadata:
                            st.json(source.metadata)
                        st.divider()
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    🚀 Built with LangChain, ChromaDB & Fine-tuned Qwen<br>
    📖 Speak with Your Book - RAG System v1.0
</div>
""", unsafe_allow_html=True)