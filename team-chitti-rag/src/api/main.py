"""FastAPI backend for Team Chitti RAG.

Provides endpoints for:
- Switching modes (local/online/hybrid) at runtime
- Saving runtime config (API keys) for the current process
- Warming up local models for low-latency chat
- Chat with/without RAG
- Building a local FAISS index from precomputed chunks

Notes:
- For security: do not persist API keys to disk here.
  The frontend may store keys in browser storage and send them to this API.
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config.mode_selector import mode_selector, ComponentType
from ..config.hybrid_config import hybrid_config
from ..embeddings.embeddings_factory import get_embeddings, clear_embeddings_cache
from ..llm.llm_factory import get_llm, reset_llm_cache
from ..vectorstore.faiss_vectorstore import FAISSVectorStore
from ..vectorstore.vectorstore_factory import reset_vectorstore_cache


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_JSONL = PROJECT_ROOT / "data" / "chunks" / "sample_docs.chunks.jsonl"


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    denom = float(np.linalg.norm(vec))
    if denom == 0.0:
        return vec
    return (vec / denom).astype(np.float32)


def _apply_env_updates(updates: Dict[str, str | None]) -> None:
    for key, value in updates.items():
        if value is None:
            # Do not remove env by default; treat None as "ignore"
            continue
        os.environ[key] = str(value)


# Keep a single FAISS index instance for low-latency retrieval.
_faiss_index: Optional[FAISSVectorStore] = None
_faiss_index_path: Optional[str] = None


class ConfigUpdateRequest(BaseModel):
    rag_mode: Optional[str] = Field(default=None, description="local|online|hybrid")

    # Hybrid overrides (optional)
    rag_llm: Optional[str] = None
    rag_embeddings: Optional[str] = None
    rag_vectorstore: Optional[str] = None
    rag_database: Optional[str] = None

    # Online config (optional)
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_api_version: Optional[str] = None
    azure_openai_deployment: Optional[str] = None
    azure_openai_model: Optional[str] = None

    openai_api_key: Optional[str] = None
    openai_embedding_model: Optional[str] = None

    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection: Optional[str] = None

    mongodb_atlas_uri: Optional[str] = None


class WarmupResponse(BaseModel):
    ok: bool
    llm: Dict[str, Any]


class ChatRequest(BaseModel):
    message: str
    # None => auto (server decides). True => force RAG. False => force no-RAG.
    use_rag: Optional[bool] = None
    top_k: int = 5
    max_tokens: int = 220


class ChatResponse(BaseModel):
    answer: str
    mode: Dict[str, Any]
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    # Safe UX metadata (no chain-of-thought)
    used_rag: bool = False
    timings_ms: Dict[str, float] = Field(default_factory=dict)


class BuildIndexRequest(BaseModel):
    chunks_jsonl: Optional[str] = None
    index_path: Optional[str] = None
    rebuild: bool = False
    max_chunks: int = 800


class BuildIndexResponse(BaseModel):
    ok: bool
    indexed_chunks: int
    index_path: str


app = FastAPI(title="Team Chitti RAG API", version="1.0.0")

API_PREFIX = "/api"

SARAS_SYSTEM_BASE = (
    "You are Saras, a friendly education-focused assistant. "
    "Saras is a well-known chatbot for education-related help started from Patna, Bihar and created during the Virsat Hackathon "
    "at Patna Women's College on 30 Jan at 2:57pm. "
    "Built by BCA 2nd-year students from the College of Commerce, Arts and Science: Sumit, Gautam, Sachin, Aditya, Anshu. "
    "This was the team's first hackathon.\n\n"
    "Primary purpose: help admins/staff organize, filter, and manage bulk student/academic data in a private environment "
    "(bootable USB, Codespaces, VM, or local computer). "
    "You can answer common user queries beyond documents when appropriate, but regularly remind users that Saras is optimized for "
    "bulk data organization/filtering/management for students and academic purposes. "
    "Be concise, helpful, and conversational. "
    "Do NOT repeat your origin story unless the user asks who you are, who built you, or your background."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)


@app.on_event("startup")
def _startup() -> None:
    # Opt-in warm startup to keep latency low.
    # Controlled via env vars so Codespaces users can decide when to pay the cost.
    auto_warmup = os.getenv("AUTO_WARMUP", "false").lower() in {"1", "true", "yes"}
    auto_load_index = os.getenv("AUTO_LOAD_INDEX", "false").lower() in {"1", "true", "yes"}

    if auto_warmup:
        try:
            llm = get_llm()
            _ = llm.generate("Say 'ready'.", max_tokens=8)
        except Exception:
            # Don't fail server startup if warmup fails.
            pass

    if auto_load_index:
        try:
            index_path = _resolve_faiss_index_path()
            _load_or_build_faiss(index_path)
        except Exception:
            pass


@app.get(f"{API_PREFIX}/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "service": "Team Chitti RAG API",
        "hint": "Use /api/* endpoints (e.g. /api/health).",
    }


# Backward-compatible aliases (older clients/scripts may call these)
@app.get("/health")
def health_alias() -> Dict[str, Any]:
    return health()


@app.get(f"{API_PREFIX}/status")
def status() -> Dict[str, Any]:
    return {
        "mode": mode_selector.get_mode_info(),
        "components": {
            "llm": mode_selector.get_component_choice(ComponentType.LLM),
            "embeddings": mode_selector.get_component_choice(ComponentType.EMBEDDINGS),
            "vectorstore": mode_selector.get_component_choice(ComponentType.VECTORSTORE),
            "database": mode_selector.get_component_choice(ComponentType.DATABASE),
        },
        "faiss": {
            "loaded": _faiss_index is not None,
            "index_path": _faiss_index_path,
            "vector_count": _faiss_index.get_vector_count() if _faiss_index else 0,
        },
    }


@app.get("/status")
def status_alias() -> Dict[str, Any]:
    return status()


@app.post(f"{API_PREFIX}/config/apply")
def apply_config(req: ConfigUpdateRequest) -> Dict[str, Any]:
    # Map request fields to env vars used by configs
    updates: Dict[str, str | None] = {}

    if req.rag_mode:
        updates["RAG_MODE"] = req.rag_mode

    if req.rag_llm is not None:
        updates["RAG_LLM"] = req.rag_llm
    if req.rag_embeddings is not None:
        updates["RAG_EMBEDDINGS"] = req.rag_embeddings
    if req.rag_vectorstore is not None:
        updates["RAG_VECTORSTORE"] = req.rag_vectorstore
    if req.rag_database is not None:
        updates["RAG_DATABASE"] = req.rag_database

    # Azure
    if req.azure_openai_endpoint is not None:
        updates["AZURE_OPENAI_ENDPOINT"] = req.azure_openai_endpoint
    if req.azure_openai_api_key is not None:
        updates["AZURE_OPENAI_API_KEY"] = req.azure_openai_api_key
    if req.azure_openai_api_version is not None:
        updates["AZURE_OPENAI_API_VERSION"] = req.azure_openai_api_version
    if req.azure_openai_deployment is not None:
        updates["AZURE_OPENAI_DEPLOYMENT"] = req.azure_openai_deployment
    if req.azure_openai_model is not None:
        updates["AZURE_OPENAI_MODEL"] = req.azure_openai_model

    # OpenAI
    if req.openai_api_key is not None:
        updates["OPENAI_API_KEY"] = req.openai_api_key
    if req.openai_embedding_model is not None:
        updates["OPENAI_EMBEDDING_MODEL"] = req.openai_embedding_model

    # Qdrant
    if req.qdrant_url is not None:
        updates["QDRANT_URL"] = req.qdrant_url
    if req.qdrant_api_key is not None:
        updates["QDRANT_API_KEY"] = req.qdrant_api_key
    if req.qdrant_collection is not None:
        updates["QDRANT_COLLECTION"] = req.qdrant_collection

    # Mongo
    if req.mongodb_atlas_uri is not None:
        updates["MONGODB_ATLAS_URI"] = req.mongodb_atlas_uri

    _apply_env_updates(updates)

    # Reload selectors/configs and reset caches so new config takes effect.
    mode_selector.reload_from_env()
    hybrid_config.reload()
    reset_llm_cache()
    clear_embeddings_cache()
    reset_vectorstore_cache()

    # Reset FAISS handle (it may point at old index/config)
    global _faiss_index, _faiss_index_path
    _faiss_index = None
    _faiss_index_path = None

    return {"ok": True, "mode": mode_selector.get_mode_info()}


@app.post(f"{API_PREFIX}/warmup", response_model=WarmupResponse)
def warmup() -> WarmupResponse:
    try:
        llm = get_llm()
        # Do a tiny generation to ensure weights are materialized
        _ = llm.generate("Say 'ready'.", max_tokens=16)
        return WarmupResponse(ok=True, llm=llm.get_model_info())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/warmup", response_model=WarmupResponse)
def warmup_alias() -> WarmupResponse:
    return warmup()


def _load_or_build_faiss(index_path: str) -> FAISSVectorStore:
    global _faiss_index, _faiss_index_path

    if _faiss_index is not None and _faiss_index_path == index_path:
        return _faiss_index

    _faiss_index = FAISSVectorStore({
        "index_path": index_path,
        "index_type": "Flat",
        "nlist": 100,
    })
    _faiss_index_path = index_path
    return _faiss_index


def _resolve_faiss_index_path() -> str:
    cfg = get_local_config_safe()
    index_path = cfg.get("faiss_index_path") or os.getenv("FAISS_INDEX_PATH") or "./data/vector_store/faiss_index.faiss"
    if not str(index_path).endswith(".faiss"):
        index_path = str(index_path) + ".faiss"
    return str(index_path)


def _index_file_exists(index_path: str) -> bool:
    try:
        return Path(index_path).expanduser().exists()
    except Exception:
        return False


_TRIVIAL_PATTERNS = {
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "who are you",
    "what can you do",
}


def _normalize_user_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_trivial_message(text: str) -> bool:
    norm = _normalize_user_text(text)
    if not norm:
        return True

    if norm in _TRIVIAL_PATTERNS:
        return True

    # Very short / low-signal messages should not trigger retrieval
    tokens = norm.split()
    if len(tokens) <= 2 and len(norm) <= 8:
        return True

    return False


def _rag_min_score() -> float:
    # FAISS scores are inner-product; since we L2-normalize vectors, this is cosine similarity.
    # 0.55 is a conservative default to avoid irrelevant chunks for short queries.
    try:
        return float(os.getenv("RAG_MIN_SCORE", "0.55"))
    except Exception:
        return 0.55


def _auto_use_rag(message: str) -> bool:
    """Heuristic router for auto RAG.

    - Skip RAG for greetings/small-talk (trivial messages)
    - Prefer RAG for question-like or longer queries
    """
    if _is_trivial_message(message):
        return False

    # If there's no index at all, don't try retrieval in auto mode.
    # (Forced RAG still errors with a clear message.)
    index_path = _resolve_faiss_index_path()
    if not _index_file_exists(index_path):
        return False

    msg = (message or "").strip()
    if "?" in msg:
        return True

    # Longer queries are usually information-seeking.
    if len(msg.split()) >= 8:
        return True

    return True


@app.post(f"{API_PREFIX}/rag/index/build", response_model=BuildIndexResponse)
def rag_build_index(req: BuildIndexRequest = Body(default=BuildIndexRequest())) -> BuildIndexResponse:
    chunks_path = Path(req.chunks_jsonl) if req.chunks_jsonl else DEFAULT_CHUNKS_JSONL
    if not chunks_path.is_absolute():
        chunks_path = PROJECT_ROOT / chunks_path

    if not chunks_path.exists():
        raise HTTPException(status_code=400, detail=f"Chunks JSONL not found: {chunks_path}")

    cfg = hybrid_config.get_general_config()
    index_path = req.index_path or os.getenv("FAISS_INDEX_PATH") or "./data/vector_store/faiss_index.faiss"
    if not index_path.endswith(".faiss"):
        index_path = index_path + ".faiss"

    index = _load_or_build_faiss(index_path)

    if req.rebuild:
        # Reset by dropping handle and recreating a new store
        global _faiss_index, _faiss_index_path
        _faiss_index = None
        _faiss_index_path = None
        index = _load_or_build_faiss(index_path)

    # If already populated and not rebuilding, return
    if index.get_vector_count() > 0 and not req.rebuild:
        return BuildIndexResponse(ok=True, indexed_chunks=index.get_vector_count(), index_path=index_path)

    embeddings = get_embeddings(force_reload=True)

    vectors: List[np.ndarray] = []
    metadatas: List[Dict[str, Any]] = []

    with chunks_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= req.max_chunks:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            content = (row.get("content") or "").strip()
            if not content:
                continue

            vec = _l2_normalize(embeddings.embed_text(content))
            vectors.append(vec)
            metadatas.append({
                "chunk_id": row.get("chunk_id"),
                "source_file": row.get("source_file"),
                "file_type": row.get("file_type"),
                "content": content,
                "metadata": row.get("metadata") or {},
            })

    index.add_vectors(vectors, metadatas)
    index.save_index()

    clear_embeddings_cache()
    gc.collect()

    return BuildIndexResponse(ok=True, indexed_chunks=index.get_vector_count(), index_path=index_path)


@app.post("/rag/index/build", response_model=BuildIndexResponse)
def rag_build_index_alias(req: BuildIndexRequest = Body(default=BuildIndexRequest())) -> BuildIndexResponse:
    return rag_build_index(req)


@app.post(f"{API_PREFIX}/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        t0 = time.perf_counter()
        mode_info = mode_selector.get_mode_info()

        # Always handle trivial/small-talk upfront.
        if _is_trivial_message(req.message):
            norm = _normalize_user_text(req.message)
            if norm in {"who are you", "what can you do"}:
                return ChatResponse(
                    answer=(
                        "I'm Saras — an education-focused assistant built during the Virsat Hackathon at Patna Women's College. "
                        "I can chat normally, and I can also use your loaded documents (RAG) when it helps. "
                        "Reminder: I'm mainly for bulk student/academic data organization, filtering, and management."
                    ),
                    mode=mode_info,
                    sources=[],
                    used_rag=False,
                    timings_ms={"total": (time.perf_counter() - t0) * 1000.0},
                )

            return ChatResponse(
                answer=(
                    "Hi! I'm Saras. How can I help today?\n\n"
                    "I can answer general questions, and I can also use your academic documents when needed. "
                    "Reminder: I'm mainly for bulk student/academic data organization, filtering, and management."
                ),
                mode=mode_info,
                sources=[],
                used_rag=False,
                timings_ms={"total": (time.perf_counter() - t0) * 1000.0},
            )

        # Decide whether to use RAG.
        # - If client forces it, obey.
        # - Otherwise auto-route.
        use_rag = req.use_rag if req.use_rag is not None else _auto_use_rag(req.message)

        if not use_rag:
            llm = get_llm()
            t_gen0 = time.perf_counter()
            answer = llm.generate(req.message, system_message=SARAS_SYSTEM_BASE, max_tokens=req.max_tokens)
            t1 = time.perf_counter()
            return ChatResponse(
                answer=answer,
                mode=mode_info,
                sources=[],
                used_rag=False,
                timings_ms={"generation": (t1 - t_gen0) * 1000.0, "total": (t1 - t0) * 1000.0},
            )

        # With RAG
        t_retr0 = time.perf_counter()
        index_path = _resolve_faiss_index_path()
        index = _load_or_build_faiss(index_path)
        if index.get_vector_count() == 0:
            if req.use_rag is True:
                raise HTTPException(status_code=400, detail="FAISS index is empty. Build it via /api/rag/index/build")
            llm = get_llm()
            t_gen0 = time.perf_counter()
            answer = llm.generate(req.message, system_message=SARAS_SYSTEM_BASE, max_tokens=req.max_tokens)
            t1 = time.perf_counter()
            return ChatResponse(
                answer=answer,
                mode=mode_info,
                sources=[],
                used_rag=False,
                timings_ms={"retrieval": (t1 - t_retr0) * 1000.0, "generation": (t1 - t_gen0) * 1000.0, "total": (t1 - t0) * 1000.0},
            )

        embeddings = get_embeddings(force_reload=True)
        qvec = _l2_normalize(embeddings.embed_text(req.message))
        hits = index.search_with_metadata(qvec, k=req.top_k)

        t_after_retr = time.perf_counter()

        clear_embeddings_cache()
        gc.collect()

        # If retrieval quality is low, don't force a context-only answer.
        max_score = max((h.get("score") or -1.0) for h in hits) if hits else -1.0
        if max_score < _rag_min_score():
            llm = get_llm()
            t_gen0 = time.perf_counter()
            answer = llm.generate(
                req.message,
                system_message=SARAS_SYSTEM_BASE,
                max_tokens=req.max_tokens,
            )
            t1 = time.perf_counter()
            return ChatResponse(
                answer=answer,
                mode=mode_info,
                sources=[],
                used_rag=False,
                timings_ms={
                    "retrieval": (t_after_retr - t_retr0) * 1000.0,
                    "generation": (t1 - t_gen0) * 1000.0,
                    "total": (t1 - t0) * 1000.0,
                },
            )

        context_parts: List[str] = []
        for hit in hits:
            md = hit.get("metadata") or {}
            content = (md.get("content") or "").strip()
            if not content:
                continue
            src = md.get("source_file") or "unknown"
            context_parts.append(f"Source: {src}\nScore: {hit.get('score'):.4f}\n\n{content}")

        context = "\n\n---\n\n".join(context_parts)
        llm = get_llm()
        system = (
            f"{SARAS_SYSTEM_BASE}\n\n"
            "You have reference context snippets below. Use them when they are relevant and helpful. "
            "If the context doesn't contain the answer, answer normally from general knowledge. "
            "If you used the context, stay consistent with it."
        )
        prompt = f"REFERENCE CONTEXT (use if helpful):\n{context}\n\nUSER: {req.message}\n\nSARAS:"
        t_gen0 = time.perf_counter()
        answer = llm.generate(prompt, system_message=system, max_tokens=req.max_tokens)
        t1 = time.perf_counter()

        # Return sources without full content by default (frontend can show preview)
        sources = [
            {
                "source_file": (h.get("metadata") or {}).get("source_file"),
                "score": h.get("score"),
                "chunk_id": (h.get("metadata") or {}).get("chunk_id"),
            }
            for h in hits
        ]

        return ChatResponse(
            answer=answer,
            mode=mode_info,
            sources=sources,
            used_rag=True,
            timings_ms={
                "retrieval": (t_after_retr - t_retr0) * 1000.0,
                "generation": (t1 - t_gen0) * 1000.0,
                "total": (t1 - t0) * 1000.0,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat_alias(req: ChatRequest) -> ChatResponse:
    return chat(req)


def get_local_config_safe() -> Dict[str, Any]:
    """Avoid importing optional deps in environments where they aren't installed."""
    try:
        from ..config.local_config import get_local_config

        cfg = get_local_config()
        return {
            "faiss_index_path": getattr(cfg, "faiss_index_path", None),
        }
    except Exception:
        return {}
