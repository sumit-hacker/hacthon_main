# Important File Structure (What does what?)

This doc is written for hackathon submission and quick maintenance.

## Top-level layout
- `main.py`
  - Simple entrypoint wrapper (project-specific; may be used by deployment or demos)
- `requirements.txt`
  - Python dependencies for backend and scripts
- `setup.sh`
  - Setup helper for the environment
- `data/`
  - Persistent artifacts (vector store, chunk cache, etc.)
- `src/`
  - Backend application code (RAG pipeline, config, API)
- `frontend/`
  - React web UI (Vite + TypeScript)
- `scripts/`
  - Utility scripts (setup, indexing, demos, comparisons)
- `reports/`
  - Generated outputs (e.g., Q&A comparison Markdown)
- `docs/`
  - Submission documentation

## Backend (Python) — key modules
### API
- `src/api/main.py`
  - FastAPI app and routes
  - Core chat endpoint
  - Saras system prompt
  - Auto-RAG policy logic (Auto/On/Off)

If deleted:
- Backend won’t run; frontend proxy requests will fail.

### Config
- `src/config/mode_selector.py`
  - Chooses local/online/hybrid mode
- `src/config/local_config.py` / `online_config.py` / `hybrid_config.py`
  - Mode-specific parameters (models, keys, paths)

If deleted:
- Mode switching breaks; config apply endpoint may fail.

### LLM
- `src/llm/llm_factory.py`
  - Returns the active LLM implementation based on config
- `src/llm/tinyllama_local_llm.py`
  - TinyLlama local inference implementation
- `src/llm/azure_openai_llm.py`
  - Online Azure OpenAI integration (optional)

If deleted:
- Chat cannot produce answers (no model backend).

### Embeddings
- `src/embeddings/embeddings_factory.py`
  - Selects embeddings implementation
- `src/embeddings/local_embeddings.py`
  - Sentence-Transformers embeddings (local)
- `src/embeddings/openai_embeddings.py`
  - OpenAI embeddings (optional)

If deleted:
- Index build fails; RAG retrieval becomes impossible.

### Vector store
- `src/vectorstore/`
  - FAISS index logic and persistence glue

If deleted:
- You can still chat without RAG, but retrieval and sources will fail.

### Database / documents
- `src/database/document_processor.py`
  - Loads docs, chunks them safely
- `src/database/sample_docs/`
  - Included sample dataset used for demo + indexing

If deleted:
- Index build may succeed only if you point to a different dataset.

## Frontend (React)
- `frontend/src/App.tsx`
  - Main UI: settings panel + chat
- `frontend/src/lib/api.ts`
  - API client wrapper
- `frontend/vite.config.ts`
  - Dev proxy (`/api` → backend)

If deleted:
- Backend still works, but there’s no web UI.

## Scripts (operational)
- `scripts/local_setup.py`
  - Local setup helper (downloads/caches models where possible, sanity checks)
- `scripts/precompute_chunks.py`
  - Precompute chunk files for faster indexing
- `scripts/demo_local_rag.py`
  - CLI demo run
- `scripts/ask_compare.py`
  - Generates “no-RAG vs RAG” comparison report

If deleted:
- Core backend still can run, but demoing/building artifacts becomes harder.

## Runtime dependencies (what relies on what)
- RAG depends on: `document_processor` → embeddings → FAISS vector store → chat pipeline
- Chat without RAG depends on: LLM only
- Frontend depends on: backend routes + CORS + `/api` prefix

## What breaks most easily
- Missing index files under `data/` → Auto mode will avoid RAG; forced RAG will error until you build index.
- Missing embeddings model cache in offline mode → embeddings load can fail; run warmup/setup with network once.

