# Setup (Local / Online / Hybrid)

This setup is designed so you can unzip the folder and run it easily.

## Quick start scripts (recommended)

First time (installs deps + starts both servers):

```bash
./run.sh --mode local
```

Daily start (no installs, just starts servers):

```bash
./start.sh
```

## 1) Prerequisites
- Python 3.10+ recommended
- Node.js 18+ recommended

## 2) Backend install

```bash
cd backend
python -m pip install -r requirements.txt
# or exact versions:
python -m pip install -r requirements.lock.txt
```

## 3) Frontend install

```bash
cd ../frontend
npm ci
```

## 4) Environment configuration

Copy env example:

```bash
cd ../backend
cp .env.example .env
```

Defaults are already `RAG_MODE=local`.

### Local mode (no keys)
- Keep `RAG_MODE=local`
- No keys required

### Online mode (needs keys)
- Set `RAG_MODE=online`
- Fill:
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_DEPLOYMENT_NAME` (or deployment)
  - `OPENAI_API_KEY` (if used)

### Hybrid mode
- Set `RAG_MODE=hybrid`
- Set component overrides (see `.env.example`)

## 5) Run

Backend (Terminal A):

```bash
cd backend
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend (Terminal B):

```bash
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Open:
- Frontend: `http://localhost:5173`
- Backend health: `http://localhost:8000/api/health`

## 6) First-time warmup + index
In the UI:
- Click **Warmup**
- Click **Build/Load Index**

Or via curl:
- `curl -X POST http://localhost:8000/api/warmup`
- `curl -X POST http://localhost:8000/api/rag/index/build`
