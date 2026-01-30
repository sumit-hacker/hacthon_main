# Warmup Keypoints + Demo Notes

## What “warmup” means here
Warmup = running a lightweight inference (and sometimes embeddings init) once so that:
- model weights are loaded into memory
- tokenization caches are created
- first-response latency is reduced

In Codespaces / CPU-only environments, the first request is often the slowest.

## Warmup checklist
- Start backend
- Call warmup endpoint from UI
- Build/load the index (if you want RAG)

Suggested order:
1) Warmup
2) Build index
3) Chat

## Important notes (practical)
- If you change modes (local/online/hybrid), warmup again.
- If you change embedding model, rebuild the index.
- If you update sample docs, rebuild the index.

## RAG policy: Auto / On / Off
- **Auto**: Saras decides if RAG is needed (best for real usage)
- **On**: Always retrieve and cite sources (best for demos)
- **Off**: Never retrieve (best for speed / casual chat)

Auto-mode behavior (simplified):
- greetings/small-talk → no RAG
- if index missing → no RAG
- if retrieval score is weak → no RAG
- else → use RAG

## Sample demo questions (good for judges)
### Non-RAG (small talk / capability)
- “Hi Saras!”
- “Who are you?”
- “What can you do for college admins?”

### RAG (dataset-grounded)
(After building index)
- “Summarize NEP 2020 in 5 points.”
- “What are the key challenges in Indian education mentioned in the documents?”
- “List the main statistics mentioned in the education statistics CSV.”
- “What does the higher education overview say about enrollment?”

### Admin-style tasks (future-facing demo)
- “Create a checklist for verifying student records.”
- “Draft a short notice for students about document verification.”

## Common problems + quick fixes
- Warmup is slow the first time → wait; it’s expected in CPU environments.
- RAG answers have no sources → you likely didn’t build/load the index.
- RAG answers look irrelevant → try increasing top_k or rebuild index after re-chunking.
- Embeddings model fails to load offline → run once with network or adjust offline setting.

## Useful commands (local dev)
From the repo root:
- Backend: `python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`
- Frontend: `cd frontend && npm install && npm run dev -- --host 0.0.0.0 --port 5173`

