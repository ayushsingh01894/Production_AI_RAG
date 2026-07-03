"""
Production AI RAG Assistant — Streamlit web interface.
Run: streamlit run streamlit_app.py

Uses the exact same rag/pipeline.py backend as app.py (CLI) — no logic
duplication between frontends.
"""

import os
import tempfile

import streamlit as st

from config import settings
from rag.pipeline import RAGPipeline

st.set_page_config(page_title="Production AI RAG Assistant", page_icon="📚", layout="wide")


# ---------- Pipeline init (cached across reruns) ----------

@st.cache_resource(show_spinner="Initializing RAG pipeline...")
def load_pipeline(llm_provider: str):
    return RAGPipeline(llm_provider=llm_provider)


# ---------- Session state ----------

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str, "sources": [...]}]
if "namespace" not in st.session_state:
    st.session_state.namespace = settings.retrieval.default_namespace


# ---------- Sidebar ----------

with st.sidebar:
    st.title("📚 RAG Assistant")

    llm_provider = st.selectbox(
        "LLM Provider", ["gemini", "openai", "groq", "ollama"],
        index=["gemini", "openai", "groq", "ollama"].index(settings.llm.provider)
        if settings.llm.provider in ["gemini", "openai", "groq", "ollama"] else 0,
    )

    st.session_state.namespace = st.text_input("Namespace", value=st.session_state.namespace)

    st.divider()
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, DOCX, or Markdown", type=["pdf", "txt", "docx", "md"], accept_multiple_files=True
    )

    if st.button("Upload & Index", use_container_width=True, disabled=not uploaded_files):
        pipeline = load_pipeline(llm_provider)
        with st.spinner("Processing documents..."):
            for uf in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uf.name)[1]) as tmp:
                    tmp.write(uf.getbuffer())
                    tmp_path = tmp.name
                try:
                    count = pipeline.upload_pdf(tmp_path, namespace=st.session_state.namespace)
                    st.success(f"{uf.name}: {count} chunks indexed.")
                except Exception as e:
                    st.error(f"{uf.name}: {e}")
                finally:
                    os.unlink(tmp_path)

    st.divider()
    st.subheader("Retrieval Settings")
    top_k = st.slider("Top-K", 1, 15, settings.retrieval.top_k)
    score_threshold = st.slider("Similarity Threshold", 0.0, 1.0, settings.retrieval.similarity_threshold, 0.05)
    mode = st.selectbox(
        "Answer Style",
        ["default", "teacher", "interviewer", "technical_expert", "beginner", "summarizer"],
    )

    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        try:
            load_pipeline(llm_provider).clear_chat_history()
        except Exception:
            pass
        st.rerun()


# ---------- Main tabs ----------

tab_chat, tab_docs, tab_stats, tab_settings = st.tabs(["💬 Chat", "📄 Documents", "📊 Statistics", "⚙️ Settings"])

# --- Chat tab ---
with tab_chat:
    st.caption(f"Namespace: `{st.session_state.namespace}` · Provider: `{llm_provider}`")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        page_info = f" · page {s.page}" if s.page else ""
                        st.markdown(f"**{s.source}**{page_info} — similarity `{s.score:.2f}`")
                        st.caption(s.text[:300] + "...")

    question = st.chat_input("Ask a question about your documents...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            pipeline = load_pipeline(llm_provider)
        except Exception as e:
            st.error(f"Pipeline initialization failed: {e}")
            st.stop()

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_answer = ""
            sources = []
            try:
                for token, chunks in pipeline.ask_stream(
                    question, namespace=st.session_state.namespace, top_k=top_k, mode=mode
                ):
                    full_answer += token
                    sources = chunks
                    placeholder.markdown(full_answer + "▌")
                placeholder.markdown(full_answer)

                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            page_info = f" · page {s.page}" if s.page else ""
                            st.markdown(f"**{s.source}**{page_info} — similarity `{s.score:.2f}`")
                            st.caption(s.text[:300] + "...")

            except Exception as e:
                full_answer = f"Error generating answer: {e}"
                placeholder.error(full_answer)
                sources = []

        st.session_state.messages.append(
            {"role": "assistant", "content": full_answer, "sources": sources}
        )

# --- Documents tab ---
with tab_docs:
    st.subheader("Indexed Namespaces")
    try:
        pipeline = load_pipeline(llm_provider)
        stats = pipeline.get_statistics()
        if stats["namespaces"]:
            for ns, count in stats["namespaces"].items():
                col1, col2, col3 = st.columns([3, 2, 2])
                col1.markdown(f"**{ns}**")
                col2.markdown(f"{count} vectors")
                if col3.button("Delete", key=f"del_{ns}"):
                    pipeline.delete_namespace(ns)
                    st.rerun()
        else:
            st.info("No documents indexed yet. Upload some from the sidebar.")
    except Exception as e:
        st.warning(f"Could not load namespaces: {e}")

    st.divider()
    st.subheader("Delete a Specific Document")
    source_to_delete = st.text_input("Source filename (as shown in citations)")
    if st.button("Delete Document") and source_to_delete:
        try:
            pipeline = load_pipeline(llm_provider)
            pipeline.delete_document(source_to_delete, namespace=st.session_state.namespace)
            st.success(f"Deleted '{source_to_delete}'.")
        except Exception as e:
            st.error(str(e))

# --- Statistics tab ---
with tab_stats:
    try:
        pipeline = load_pipeline(llm_provider)
        stats = pipeline.get_statistics()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Vectors", stats["total_vectors"])
        c2.metric("Index Dimension", stats["dimension"])
        c3.metric("Embedding Model", stats["embedding_model"])
        st.json(stats["namespaces"])
    except Exception as e:
        st.warning(f"Could not load statistics: {e}")

# --- Settings tab ---
with tab_settings:
    st.subheader("Current Configuration")
    st.json({
        "llm_provider": llm_provider,
        "embedding_model": settings.embedding.model_name,
        "chunk_size": settings.chunking.chunk_size,
        "chunk_overlap": settings.chunking.chunk_overlap,
        "top_k": top_k,
        "similarity_threshold": score_threshold,
        "namespace": st.session_state.namespace,
        "temperature": settings.llm.temperature,
        "max_tokens": settings.llm.max_tokens,
    })
    st.caption("To change API keys or defaults, edit your `.env` file and restart the app.")
