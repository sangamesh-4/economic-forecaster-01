import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# ===============================
# 🔐 LOAD ENV
# ===============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ===============================
# 📦 IMPORTS
# ===============================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FakeEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma

# ===============================
# 🎨 PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Bengaluru Hybrid Economic Intelligence AI",
    page_icon="🏦",
    layout="wide"
)

# ===============================
# 🏷 HEADER
# ===============================
st.title("🏦 Bengaluru Hybrid Economic Intelligence AI")
st.caption("Understand RBI policies, inflation trends, and economic signals using AI")

# ===============================
# 💬 SESSION STATE
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_query" not in st.session_state:
    st.session_state.selected_query = ""

# ===============================
# 📚 VECTOR STORE
# ===============================
@st.cache_resource
def get_vectorstore():
    data_path = Path("rbi_data")

    if not data_path.exists():
        return None

    docs = []

    for pdf in data_path.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {pdf}: {e}")

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = splitter.split_documents(docs)

    embeddings = FakeEmbeddings(size=384)

    vectorstore = Chroma.from_documents(splits, embeddings)

    return vectorstore

# ===============================
# 🧠 RAG QUERY ENGINE
# ===============================
def rag_query(query, k=5):

    if not GROQ_API_KEY:
        return "❌ Missing GROQ API Key.", []

    vectorstore = get_vectorstore()

    if not vectorstore:
        return "📂 No data available. Please add PDFs.", []

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )

    prompt = f"""
You are a senior economic intelligence analyst.

Use the provided context to generate a high-quality answer.

Context:
{context}

Question:
{query}

Instructions:
- Be concise but insightful
- Avoid generic responses
- Focus on economic reasoning
- Use structured format

Format:

### Key Takeaway
(Direct answer)

### Insights
- Bullet points

### Analysis
(Explain economic meaning)

### Conclusion
(Strategic implication)
"""

    response = llm.invoke(prompt)

    return response.content, docs

# ===============================
# 🧠 INTRO SECTION (TOP UX)
# ===============================
st.markdown("### 💡 What can this AI do?")
st.markdown("""
- Analyze RBI reports and policies  
- Explain inflation, repo rate, GDP trends  
- Provide structured economic insights  
- Answer finance-related questions intelligently  
""")

st.markdown("### 🚀 Try asking:")

example_queries = [
    "Impact of repo rate on inflation",
    "RBI monetary policy analysis",
    "Key drivers of GDP growth",
    "How does inflation affect economy?",
    "What is CRR and its impact?"
]

cols = st.columns(len(example_queries))

for i, q in enumerate(example_queries):
    if cols[i].button(q):
        st.session_state.selected_query = q

# ===============================
# 🎯 USER INPUT
# ===============================
query = st.text_input(
    "🔍 Ask your question:",
    value=st.session_state.selected_query,
    placeholder="Type or click an example above..."
)

# ===============================
# 🚀 PROCESS QUERY
# ===============================
if query:

    st.markdown(f"### 🔎 Your Query: {query}")

    with st.spinner("Analyzing economic data..."):
        answer, docs = rag_query(query)

    st.markdown("### 🤖 AI Insight")
    st.markdown(answer)

    st.session_state.chat_history.append({
        "query": query,
        "answer": answer
    })

# ===============================
# 📜 CHAT HISTORY
# ===============================
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 💬 Conversation History")

    for chat in reversed(st.session_state.chat_history[-5:]):
        st.markdown(f"**Q:** {chat['query']}")
        st.markdown(f"**A:** {chat['answer']}")
        st.markdown("---")
