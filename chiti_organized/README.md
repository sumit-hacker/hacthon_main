# Saras (Team Chitti) — Organized Project (Zip-and-Run)

This folder is an **organized, modular** version of the Saras RAG project.

Goal: You can **zip `chiti_organized/`**, move it anywhere, install dependencies, and run Saras in:
- **Local mode** (no API keys): TinyLlama + Local Embeddings + FAISS
- **Online mode** (with API keys): Azure/OpenAI
- **Hybrid mode** (mix components)

## Folder structure (top-level)
- `backend/` — Python/FastAPI API + RAG pipeline
- `frontend/` — React/Vite UI
- `docs/` — Saras docs + CMS references + PPT prompts
- `reports/` — generated reports and cleanup report
- `scripts/` — optional helper scripts (currently minimal)

Start here:
- Setup: see [SETUP.md](SETUP.md)
- Usage and modes: see [GUIDE.md](GUIDE.md)
