import streamlit as st
import torch
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configuration
class RAGConfig:
    """Configuration for the RAG System"""
    # Paths
    DATABASE_PATH = os.path.join(project_root, "outputs/data/vector_database")
    
    # Model IDs
    EMBEDDING_MODEL = "Mo7amed3twan/GATE-AraBert-v2"
    LLM_MODEL = "Mo7amed3twan/GradModel-2B-RAG"
    
    # Model settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_LENGTH = 512
    TEMPERATURE = 0.5
    
    # Retrieval settings
    RETRIEVAL_K = 2
    CHUNK_SIZE = 1024

class ArabicChatBot:
    def __init__(self, rag_chain, vectorstore):
        self.rag_chain = rag_chain
        self.vectorstore = vectorstore

    def ask_question(self, question: str):
        if not self.rag_chain:
            return "Error Loading Rag System (Speak With Your Book)"

        try:
            response = self.rag_chain({"query": question})
            answer = response["result"]
            sources = response["source_documents"]

            return {
                "answer": answer,
                "sources": sources,
                "sources_count": len(sources)
            }
        except Exception as e:
            return f"Error In The Question: {e}"

def load_arabic_embedding_model(config):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': config.DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        st.error(f"Error In Loading The Embedding Model: {e}")
        return None

def load_arabic_llm(config):
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL)

        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.MAX_LENGTH,
            temperature=config.TEMPERATURE,
            do_sample=True,
            return_full_text=False
        )

        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        return llm
    except Exception as e:
        st.error(f"Error In Loading The LLM: {e}")
        return None

def load_vector_database(config, embeddings):
    try:
        vectorstore = Chroma(
            persist_directory=config.DATABASE_PATH,
            embedding_function=embeddings,
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error in loading your database: {e}")
        return None

def create_rag_chain(llm, vectorstore):
    arabic_prompt_template = """
أنت مساعد ذكي صممه محمد عطوان لمشروع تخرجه (تحدث مع كتابك) تتحدث العربية الفصحى فقط. مهمتك هي الإجابة على الأسئلة بناءً على السياق المقدم فقط.

السياق:
{context}

السؤال: {question}

التعليمات:
- أجب باللغة العربية الفصحى
- استخدم المعلومات من السياق فقط للإجابة
- إذا لم تجد الإجابة في السياق، قل "لا أعرف" بدلاً من اختلاق إجابة
- كن دقيقاً وواضحاً في إجابتك

الإجابة:
"""

    PROMPT = PromptTemplate(
        template=arabic_prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": RAGConfig.RETRIEVAL_K}
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    return rag_chain

# Streamlit UI
def main():
    st.set_page_config(
        page_title="تحدث مع كتابك - محمد عطوان",
        page_icon="📚",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .css-1y4p8pa {
        direction: rtl;
    }
    .stTextInput {
        direction: rtl;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🤖 تحدث مع كتابك")
    st.markdown("##### نظام ذكي للتحدث مع المستندات العربية")

    # Initialize session state
    if 'chatbot' not in st.session_state:
        config = RAGConfig()
        embeddings = load_arabic_embedding_model(config)
        llm = load_arabic_llm(config)
        vectorstore = load_vector_database(config, embeddings)
        
        if all([embeddings, llm, vectorstore]):
            rag_chain = create_rag_chain(llm, vectorstore)
            st.session_state.chatbot = ArabicChatBot(rag_chain, vectorstore)
            st.success("✅ تم تحميل النظام بنجاح")
        else:
            st.error("❌ حدث خطأ في تحميل النظام")
            return

    # Chat interface
    with st.form("chat_form"):
        user_question = st.text_input("اكتب سؤالك هنا:", placeholder="مثال: ما هي حقوق المرأة في العمل؟")
        submit_button = st.form_submit_button("إرسال")

    if submit_button and user_question:
        with st.spinner("جاري معالجة سؤالك..."):
            response = st.session_state.chatbot.ask_question(user_question)

            if isinstance(response, dict):
                # Display answer
                st.markdown("### الإجابة:")
                st.write(response["answer"])

                # Display sources in an expander
                with st.expander(f"📚 المصادر (تم استخدام {response['sources_count']} مصدر)"):
                    for i, source in enumerate(response["sources"], 1):
                        st.markdown(f"**المصدر {i}:**")
                        st.markdown(f"```\n{source.page_content[:300]}...\n```")
                        st.markdown("**البيانات الوصفية:**")
                        st.json(source.metadata)
            else:
                st.error(response)

    # Add footer
    st.markdown("---")
    st.markdown("تم التطوير بواسطة محمد عطوان © 2025")

if __name__ == "__main__":
    main()
