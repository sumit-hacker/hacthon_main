
# Dependencies Guide (How to run Saras + Frontend)

This repo contains two parts:
- **Backend (Python/FastAPI)** in `team-chitti-rag/`
- **Frontend (React/Vite)** in `team-chitti-rag/frontend/`

This guide lists what you need to install to run the project, plus **what each dependency does**, **why it’s needed**, and **how it works**.

> Note on versions:
> - Python dependencies are declared in `team-chitti-rag/requirements.txt` using `>=` (minimum versions).
> - To reinstall **the exact same versions** used in this workspace, use `team-chitti-rag/requirements.lock.txt` (generated from `pip freeze`).
> - Frontend dependencies are pinned exactly via `team-chitti-rag/frontend/package-lock.json` (use `npm ci`).

## System prerequisites

### Python
- **Version**: Python 3.10+ recommended
- **Why**: Runs FastAPI server, RAG pipeline scripts, and local LLM integrations.
- **How it works**: Python hosts the API + model pipeline, and serves JSON responses consumed by the frontend.

Install (Ubuntu):
- `sudo apt-get update && sudo apt-get install -y python3 python3-pip`

### Node.js
- **Version**: Node 18+ recommended
- **Why**: Builds and runs the React frontend.
- **How it works**: Vite runs a dev server on port 5173 and proxies `/api/*` calls to the backend.

Exact versions used here (captured):
- `team-chitti-rag/frontend/NODE_VERSION.txt`
- `team-chitti-rag/frontend/NPM_VERSION.txt`

## Backend (Python) dependencies

Source of truth:
- `team-chitti-rag/requirements.txt`

### Core API

- **fastapi** (>=0.112.0)
	- **Why needed**: Defines the REST API (`/api/health`, `/api/chat`, `/api/warmup`, `/api/rag/index/build`).
	- **What it does**: Handles HTTP routing, request parsing, validation, and JSON responses.
	- **How it works**: Decorators like `@app.get()` and `@app.post()` register routes; Pydantic models validate request/response schemas.

- **uvicorn[standard]** (>=0.30.0)
	- **Why needed**: ASGI server used to run FastAPI.
	- **What it does**: Starts an async web server, handles concurrency and request lifecycle.
	- **How it works**: Runs `src.api.main:app` and serves it at `http://0.0.0.0:8000`.

- **pydantic** (>=2.8.0) and **pydantic-settings** (>=2.4.0)
	- **Why needed**: Configuration and request/response validation.
	- **What it does**: Type-safe models for API payloads and environment-based settings.
	- **How it works**: Pydantic validates inputs and casts types; pydantic-settings loads values from env and `.env`.

- **python-dotenv** (>=1.0.1)
	- **Why needed**: Loads `.env` values during local dev.
	- **What it does**: Reads environment variables from `.env` file.
	- **How it works**: Populates `os.environ` so configs can read keys like `RAG_MODE`.

### Local LLM (TinyLlama) + local inference

- **torch** (>=2.3.0)
	- **Why needed**: Runs the neural network model locally.
	- **What it does**: Tensor operations and model execution (CPU or GPU).
	- **How it works**: Transformers uses Torch to load weights and perform forward passes during generation.

- **transformers** (>=4.42.0)
	- **Why needed**: Loads TinyLlama model/tokenizer and performs text generation.
	- **What it does**: High-level APIs for pretrained LLMs.
	- **How it works**: Downloads/caches model weights, tokenizes input, runs generation loop.

- **accelerate** (>=0.32.0)
	- **Why needed**: Helps run HF models efficiently in different environments.
	- **What it does**: Device placement, performance helpers.
	- **How it works**: Integrates with Transformers for smoother CPU/GPU execution.

### Embeddings (local)

- **sentence-transformers** (>=3.0.0)
	- **Why needed**: Converts text chunks into numeric vectors for retrieval.
	- **What it does**: Provides `SentenceTransformer` models for embeddings.
	- **How it works**: Encodes text into fixed-size vectors (e.g., 384 dims for MiniLM).

### Vector store (FAISS)

- **faiss-cpu** (>=1.8.0)
	- **Why needed**: Fast similarity search over embedding vectors.
	- **What it does**: Indexes vectors and returns nearest matches.
	- **How it works**: Stores vectors in an index; queries via cosine similarity (often using normalized vectors + inner product).

### Data science utilities

- **numpy** (>=2.0.0)
	- **Why needed**: Vector math and array manipulation.
	- **What it does**: Fast numerical operations.
	- **How it works**: Used for normalization and embedding vector handling.

- **scipy** (>=1.14.0), **scikit-learn** (>=1.5.0)
	- **Why needed**: Some embedding/vector utilities and possible future ranking helpers.
	- **What it does**: Scientific computing + ML utilities.
	- **How it works**: May be used indirectly by sentence-transformers.

### Document parsing + ingestion

- **pypdf** (>=4.3.0)
	- **Why needed**: Read text from PDF documents.
	- **How it works**: Parses PDF pages and extracts text.

- **python-docx** (>=1.2.0)
	- **Why needed**: Read text from DOCX files.
	- **How it works**: Parses Office XML and extracts paragraphs.

- **markdown** (>=3.7.0)
	- **Why needed**: Handles `.md` content.
	- **How it works**: Converts or normalizes markdown content for chunking.

- **beautifulsoup4** (>=4.13.0)
	- **Why needed**: HTML parsing when needed.
	- **How it works**: Extracts text from HTML content.

- **python-multipart** (>=0.0.9)
	- **Why needed**: File uploads in FastAPI (attachments, if enabled).
	- **How it works**: Parses multipart/form-data requests.

### Optional cloud/online mode (only if you switch `RAG_MODE=online`)

- **openai** (>=1.35.0)
	- **Why needed**: If using OpenAI/Azure OpenAI APIs.
	- **How it works**: Client SDK sends HTTPS requests with your API key.

- **qdrant-client** (>=1.9.0)
	- **Why needed**: If using Qdrant vector DB in online mode.
	- **How it works**: Inserts/searches vectors via Qdrant HTTP/gRPC APIs.

- **pymongo** (>=4.8.0), **motor** (>=3.5.0)
	- **Why needed**: If using MongoDB (local or Atlas).
	- **How it works**: MongoDB client libraries for storing/retrieving metadata.

## Frontend (Node/React) dependencies

Source of truth:
- `team-chitti-rag/frontend/package.json`
- `team-chitti-rag/frontend/package-lock.json` (exact pinned versions)

Runtime:
- **react** (^19.2.0) + **react-dom** (^19.2.0)
	- **Why needed**: UI rendering and component state.
	- **How it works**: React renders the chat + settings panel; calls backend via fetch.

Build/dev:
- **vite** (^7.2.4)
	- **Why needed**: Dev server + build tool.
	- **How it works**: Runs on 5173, hot reload, and proxies `/api` to backend.

- **typescript** (~5.9.3)
	- **Why needed**: Type safety in the frontend.

- **@vitejs/plugin-react**, **eslint** and related plugins
	- **Why needed**: React build integration + code quality.

## How to install everything again (same versions)

### Backend (exact versions)
From repo root:

- `python -m pip install -r team-chitti-rag/requirements.lock.txt`

### Backend (minimum versions)
- `python -m pip install -r team-chitti-rag/requirements.txt`

### Frontend (exact versions)
From repo root:

- `cd team-chitti-rag/frontend && npm ci`

## How to run

Backend:
- `cd team-chitti-rag && python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`

Frontend:
- `cd team-chitti-rag/frontend && npm run dev -- --host 0.0.0.0 --port 5173`

