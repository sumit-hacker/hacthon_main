# Guide (How Saras works + modes)

## What Saras does
Saras is an education-focused assistant for admin-side work (bulk student/academic data organization, filtering, and management).

It supports normal chat and RAG (document-grounded answers with sources).

## How modes work
Modes are controlled by env vars (see `backend/.env.example`).

- `local`: TinyLlama + local embeddings + FAISS (best for offline/private)
- `online`: cloud model + hosted vector/db (best quality if you have keys)
- `hybrid`: mix components

The frontend lets you choose:
- RAG policy: **Auto / On / Off**

## Where to put your documents
- Add docs under `backend/src/database/sample_docs/` (demo dataset)
- Rebuild index after adding documents:
  - UI: **Build/Load Index**
  - or API: `POST /api/rag/index/build`

## Useful endpoints
- `GET /api/health`
- `GET /api/status`
- `POST /api/warmup`
- `POST /api/rag/index/build`
- `POST /api/chat`

## Troubleshooting
- If first response is slow: run warmup.
- If RAG shows no sources: build/load index.
- If you removed `node_modules`: run `npm ci`.
- If you changed `.env`: restart backend.
