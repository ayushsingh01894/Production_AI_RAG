# Production AI RAG Assistant

A modular, production-style **Retrieval-Augmented Generation (RAG)** system built on
**Pinecone** (vector DB) + a **provider-independent LLM layer** (Gemini / OpenAI / Groq / Ollama),
with both a **CLI** and a **Streamlit web UI** sharing the same backend pipeline.

```
User → PDF/TXT/DOCX Loader → Smart Chunking → SentenceTransformer Embeddings
     → Pinecone Vector DB (namespaces + metadata) → Semantic Search
     → Context Builder → LLM (Gemini/OpenAI/Groq/Ollama) → Answer + Sources
```

## Features

- Multi-format ingestion: PDF, TXT, Markdown, DOCX (single file / multiple / folder)
- Recursive smart chunking with configurable size & overlap
- Batched SentenceTransformer embeddings (swap model via config)
- Full Pinecone CRUD: create index, upsert, update, delete, namespaces, metadata filters
- Semantic search: top-K, similarity threshold, duplicate removal, metadata filters
- Provider-independent LLM (Gemini / OpenAI / Groq / Ollama) — switch via `.env`, no code changes
- Prompt modes: default, teacher, interviewer, technical_expert, beginner, summarizer
- Source citations with similarity scores (page + filename)
- Conversation memory with configurable rolling window
- Response caching for repeated questions
- Retry with exponential backoff on API failures
- Centralized logging (console + rotating file)
- CLI console app (`app.py`) and Streamlit web app (`streamlit_app.py`) — same core pipeline, zero duplication
- Streaming token-by-token answers in the Streamlit UI

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# edit .env: add your PINECONE_API_KEY and at least one LLM provider key
```

### Get API keys
- **Pinecone**: https://app.pinecone.io
- **Gemini**: https://aistudio.google.com/app/apikey
- **OpenAI**: https://platform.openai.com/api-keys
- **Groq**: https://console.groq.com/keys
- **Ollama** (offline, no key needed): https://ollama.com — run `ollama pull llama3` locally first

## Run

**CLI:**
```bash
python app.py
```

**Streamlit Web UI:**
```bash
streamlit run streamlit_app.py
```

## Project Structure

```
Production_AI_RAG/
├── app.py                  # CLI entry point
├── streamlit_app.py        # Streamlit web UI
├── config.py                # Central settings (loaded from .env)
├── requirements.txt
├── .env.example
│
├── loaders/pdf_loader.py           # PDF/TXT/DOCX/MD loading
├── chunking/chunker.py             # Recursive smart chunker
├── embeddings/embedder.py          # SentenceTransformer wrapper
│
├── pinecone_db/
│   ├── pinecone_manager.py         # Index create/connect
│   ├── uploader.py                 # Upsert/update/delete
│   ├── search.py                   # Semantic search
│   ├── namespace_manager.py        # Namespace utilities
│   └── metadata_filter.py          # Filter builder
│
├── llm/
│   ├── llm_client.py               # Multi-provider LLM client
│   ├── prompt_builder.py           # System + context + history prompt
│   └── response_generator.py       # Answer generation + caching
│
├── rag/
│   ├── retrieval.py                # Retriever wrapper
│   └── pipeline.py                 # Full pipeline orchestration
│
├── memory/chat_history.py          # Rolling conversation memory
├── models/document.py              # Chunk / RetrievedChunk / RAGAnswer dataclasses
├── utils/logger.py, helper.py      # Logging + retry/hash helpers
└── prompts/system_prompt.txt       # Base system prompt
```

## Switching LLM Providers

Just change one line in `.env`:

```
LLM_PROVIDER=gemini    # or openai / groq / ollama
```

No code changes needed anywhere in the project.

## Switching Embedding Models

```
EMBEDDING_MODEL=all-MiniLM-L6-v2   # or bge-small, bge-base, e5-base, nomic-embed...
EMBEDDING_DIMENSION=384             # must match the chosen model's output dimension
```

⚠️ If you change the embedding model/dimension after vectors already exist in your Pinecone
index, you must create a new index (dimension is fixed per index).

## Notes

- `cache/response_cache.json` stores cached answers — delete it to force fresh generations.
- Logs are written to `logs/app.log` (rotating, 3 backups, 5MB each).
- Namespaces let you isolate different document collections (e.g. per project/client) inside one index.
