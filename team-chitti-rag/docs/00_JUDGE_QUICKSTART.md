# Judge Quickstart (3–5 minutes)

This quickstart runs Saras **fully local** (TinyLlama + local embeddings + FAISS) with **zero online API keys**.

## What you’ll see
- Saras answers normal chat questions without retrieval.
- Saras uses RAG (retrieval + sources) only when needed (Auto) or always (On).

## Prerequisites
- Python 3.10+ recommended
- Node.js 18+

Note: On the first run, models may download (TinyLlama + embeddings). This needs internet once, but **no API keys**.

## 1) Install dependencies
From `team-chitti-rag/`:

```bash
python -m pip install -r requirements.txt
```

Frontend:

```bash
cd frontend
npm install
cd ..
```

## 2) Start the backend
From `team-chitti-rag/`:

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Backend checks:
- `GET /api/health` → `{ ok: true }`

## 3) Start the frontend
In a second terminal:

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Open the forwarded port / URL for the frontend.

## 4) One-click demo flow in UI
1. Confirm **Mode = local**
   - If it’s not local (because browser storage remembered), switch it to local.
2. Click **Warmup** (first run may take time)
3. Click **Build/Load Index**
4. Set **RAG Policy**:
   - **Auto** (recommended real behavior)
   - or **On** (best for judge demo because it will always show sources)

## 5) Judge questions (copy/paste)
### A) Quick sanity (no RAG)
- “Hi Saras!”
- “Who are you?”

Expected: friendly response; no irrelevant document citations.

### B) RAG demo (after index build)
- “Summarize NEP 2020 in 5 bullet points.”
- “What does the dataset say about higher education in India?”

Expected: answer includes document-based grounding and shows sources.

### C) Admin usefulness (non-document)
- “Create a checklist for verifying student documents during admission.”

Expected: practical admin-oriented output (may not cite sources).

## Success criteria (what counts as ‘working’)
- Frontend loads and connects to backend
- Warmup completes (even if slow)
- Index builds/loads
- RAG answers show sources when RAG Policy is **On** (or Auto decides to use RAG)

## Troubleshooting (fast)
- Warmup is slow the first time → normal on CPU.
- No sources are shown → click **Build/Load Index** and try again.
- “RAG forced but index missing” → switch RAG Policy to **Auto** or build index.
- If the UI had old settings → set Mode to **local** (or clear site storage).
