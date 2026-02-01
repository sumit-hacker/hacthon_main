# Saras (Team Chitti RAG) — Project Overview

## What it is
**Saras** is a friendly, education-focused assistant built for **admin-side work**: organizing, filtering, and managing **bulk student/academic data** in a private environment.

- Built during **Virsat Hackathon** at **Patna Women’s College**, Patna, Bihar
- Date/time: **30 Jan, 2:57 PM**
- Team: BCA 2nd-year students (College of Commerce, Arts and Science)
  - **Sumit, Gautam, Sachin, Aditya, Anshu**
- First hackathon for the team

## Why Saras exists (use-case)
In colleges/schools, admins often deal with:
- Large document folders (PDF/DOCX/MD/TXT/CSV)
- Multiple sources/links and repeated data
- Need for quick answers + quick sorting (not just chat)

Saras helps by:
- Answering normal questions conversationally
- Using **RAG** (retrieval over documents) only when it’s actually helpful
- Working in private setups: **bootable USB**, **Codespaces**, **VMs**, or **local PCs**

## How we achieved it (high level)
We implemented a modular RAG pipeline where each component can be swapped:

1. **LLM** (local): TinyLlama
2. **Embeddings** (local): Sentence-Transformers
3. **Vector store** (local): FAISS
4. **API server**: FastAPI
5. **Frontend**: React (Vite)

The UI lets the user:
- Warm up the model (reduce cold-start latency)
- Build/load the index
- Chat with RAG policy set to **Auto / On / Off**

## Problems we faced (and solved)
### 1) “RAG for everything” (bad UX)
**Problem:** Even “hi” triggered retrieval, producing nonsense “context-only” answers.

**Fix:** Auto-routing + guardrails.
- Trivial/small-talk messages never force RAG
- RAG is used only when:
  - there is an index, and
  - similarity is strong enough
- Otherwise Saras answers normally.

### 2) Chunking stalls / termination
**Problem:** Chunking could stall (non-progress loop) and the process got terminated.

**Fix:** Ensured chunking always makes forward progress and safely handles overlap/boundaries.

### 3) Codespaces-friendly local model choice
**Problem:** Heavy models can be slow or fail in limited environments.

**Fix:** Defaulted local mode to TinyLlama (lightweight + CPU-friendly).

### 4) Dependency/install friction
**Problem:** Some dependencies were missing or scripts weren’t on PATH.

**Fix:**
- Use `python -m uvicorn` to run the server reliably
- Avoid embeddings network timeouts by preferring cached/local model load when possible

## Pros and cons
### Pros
- Works in **offline / private** environments
- Fast retrieval with FAISS
- “Auto RAG” = better chat experience
- Mode system supports future online/hybrid expansion
- Frontend makes it demo-ready for judges

### Cons / tradeoffs
- Local CPU inference can be slow under load
- RAG quality depends heavily on good chunking + good embeddings + enough indexed chunks
- Large indexes can increase memory usage

## What’s next (if we had more time)
- Add real admin tools: bulk tagging, duplicate detection, filters, exports
- Better retrieval ranking + reranking
- Multi-document summarization and report generation
- Auth + audit logs for admin workflows
