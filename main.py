import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# -------------------------------
# LOAD ENV
# -------------------------------
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

# -------------------------------
# IMPORTS
# -------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma

# -------------------------------
# EMBEDDINGS
# -------------------------------
from langchain_community.embeddings import FakeEmbeddings

class CustomEmbeddings:
    def __init__(self):
        self.model = FakeEmbeddings(size=384)

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.embed_query(text)

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Economic Intelligence AI", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# UI
# -------------------------------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#f8fafc,#e2e8f0);}
.user {background:#2563eb;color:white;padding:12px;border-radius:14px;margin:10px 0;}
.bot {background:white;padding:14px;border-radius:14px;border:1px solid #e2e8f0;margin:10px 0;}
.center-box {text-align:center;margin-top:12vh;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# VECTOR STORE (PERSISTENT ⚡)
# -------------------------------
@st.cache_resource
def get_vectorstore():
    persist_dir = "db"

    embeddings = CustomEmbeddings()

    # If DB already exists → load
    if Path(persist_dir).exists():
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

    # Else build once
    docs = []
    for pdf in Path("rbi_data").glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        splits,
        embeddings,
        persist_directory=persist_dir
    )

    return vectorstore

# -------------------------------
# LLM
# -------------------------------
def get_llm(temp=0.2):  # ⚡ faster
    return ChatGroq(
        temperature=temp,
        groq_api_key=groq_key,
        model_name="llama-3.1-8b-instant"
    )

# -------------------------------
# RERANKER
# -------------------------------
def rerank_docs(query, docs):
    scored = []
    for d in docs:
        score = len(set(query.lower().split()) & set(d.page_content.lower().split()))
        scored.append((score, d))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [d for _, d in scored[:4]]

# -------------------------------
# RAG PIPELINE (OPTIMIZED ⚡)
# -------------------------------
def rag_pipeline(query):

    vectorstore = get_vectorstore()
    if not vectorstore:
        return "No data available"

    # 🔥 No rewrite → faster
    better_query = query

    # 🔥 Optimized retrieval
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k":4, "fetch_k":8}
    )

    docs = retriever.invoke(better_query)

    # 🔥 Rerank
    docs = rerank_docs(better_query, docs)

    # Context
    context = "\n\n".join([d.page_content[:400] for d in docs])

    # Confidence
    confidence = "High" if len(docs)>=3 else "Medium" if len(docs)>=2 else "Low"

    # Sources
    sources = list(set([
        Path(d.metadata.get("source","Unknown")).name for d in docs
    ]))

    # Memory
    history = "\n".join([
        f"User: {c['q']}\nAI: {c['a']}"
        for c in st.session_state.chat_history[-2:]
    ])

    # LLM
    llm = get_llm(0.2)

    prompt = f"""
You are an elite economic analyst.

Conversation:
{history}

Context:
{context}

Question:
{query}

Provide structured answer:

### 📊 Key Insights
### 🧠 Analysis
### ⚠️ Implications
### 📚 Evidence
### 📈 Confidence: {confidence}
### 📌 Final Takeaway
"""

    answer = llm.invoke(prompt).content

    # Add sources
    source_text = "\n".join([f"- {s}" for s in sources])
    final_answer = answer + f"\n\n---\n### 📎 Sources\n{source_text}"

    return final_answer

# -------------------------------
# HEADER
# -------------------------------
st.title("🏦  Economic Intelligence AI")

# -------------------------------
# CHAT HISTORY
# -------------------------------
for c in st.session_state.chat_history:
    st.markdown(f"<div class='user'>{c['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot'>{c['a']}</div>", unsafe_allow_html=True)

# -------------------------------
# INPUT
# -------------------------------
query = st.chat_input("Ask about RBI, inflation, GDP...")

# -------------------------------
# MAIN FLOW
# -------------------------------
if query:

    st.markdown(f"<div class='user'>{query}</div>", unsafe_allow_html=True)

    with st.spinner("⚡ Analyzing..."):
        answer = rag_pipeline(query)

    st.session_state.chat_history.append({"q":query,"a":answer})

    st.markdown(f"<div class='bot'>{answer}</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.download_button("📥 Download Report", data=answer)

    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history=[]
            st.rerun()

# -------------------------------
# LANDING
# -------------------------------c
else:
    st.markdown("""
    <div class="center-box">
    <h2>Bengaluru Hybrid Economic Intelligence AI</h2>
    <p style='text-align:center; color:gray;'>Built by H Sangamesh</p>
    </div>
    """, unsafe_allow_html=True)

    col1,col2,col3=st.columns(3)
    col1.info("Impact of repo rate on inflation")
    col2.info("GDP growth drivers in India")
    col3.info("RBI monetary policy analysis")

