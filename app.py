"""
New Project #1 — RAG Chatbot over Your Own Documents
-------------------------------------------------------
Why add this: your CV lists LangChain + RAG as skills but no project actually
demonstrates them end-to-end. This is the single most in-demand "GenAI" project
pattern for remote ML/DS roles right now (customer-support bots, internal
knowledge assistants, doc Q&A).

What it does: upload PDFs -> chunks + embeds them -> stores in a local FAISS
vector index -> answers questions grounded only in those documents, with
source citations.

Run locally:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-...      # or OPENAI_API_KEY, see note below
    streamlit run app.py

Deploy: Streamlit Cloud or HF Spaces (add your API key as a secret, never hardcode it).
"""

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA

st.set_page_config(page_title="RAG Document Chatbot", layout="wide")
st.title("📄 RAG Chatbot — Ask Questions About Your Documents")
st.caption("Upload one or more PDFs. Answers are grounded in the documents, with sources cited.")

if "ANTHROPIC_API_KEY" not in os.environ:
    st.warning("Set the ANTHROPIC_API_KEY environment variable (or swap in OPENAI_API_KEY + "
               "`from langchain_openai import ChatOpenAI` below) before running.")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)


@st.cache_resource(show_spinner="Indexing documents...")
def build_index(file_bytes_list):
    docs = []
    for i, file_bytes in enumerate(file_bytes_list):
        tmp_path = f"/tmp/upload_{i}.pdf"
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Free, local embedding model — no API cost for the retrieval step
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


if uploaded_files:
    file_bytes_list = [f.read() for f in uploaded_files]
    vectorstore = build_index(file_bytes_list)

    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )

    query = st.text_input("Ask a question about your documents:")
    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": query})
        st.markdown("### Answer")
        st.write(result["result"])

        with st.expander("Sources used"):
            for doc in result["source_documents"]:
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Page {page}:** {doc.page_content[:300]}...")
else:
    st.info("Upload a PDF to get started — try a resume, a research paper, or a company handbook.")
