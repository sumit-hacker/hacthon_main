#!/usr/bin/env python3
"""Compare answers: TinyLlama without RAG vs with RAG.

- Without RAG: ask TinyLlama directly.
- With RAG: retrieve context from FAISS built from precomputed chunks, then ask TinyLlama.

Writes a Markdown report with both answers.

Prereqs:
- Run `python scripts/precompute_chunks.py` at least once (creates data/chunks/*.jsonl)

Usage:
    python scripts/ask_compare.py --question "What is NEP 2020?" --out reports/qa_results.md
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.local_config import get_local_config
from src.embeddings.embeddings_factory import get_embeddings, clear_embeddings_cache
from src.llm.llm_factory import get_llm
from src.vectorstore.faiss_vectorstore import FAISSVectorStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    denom = float(np.linalg.norm(vec))
    if denom == 0.0:
        return vec
    return (vec / denom).astype(np.float32)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_index_from_chunks(
    chunks_jsonl: Path,
    index_path: Path,
    rebuild: bool,
    max_chunks: int | None,
) -> FAISSVectorStore:
    index_path.parent.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISSVectorStore({
        "index_path": str(index_path),
        "index_type": "Flat",
        "nlist": 100,
    })

    if (not rebuild) and vectorstore.get_vector_count() > 0:
        return vectorstore

    embeddings = get_embeddings(force_reload=True)

    vectors: List[np.ndarray] = []
    metadatas: List[Dict[str, Any]] = []

    for i, row in enumerate(_iter_jsonl(chunks_jsonl)):
        if max_chunks is not None and i >= max_chunks:
            break

        content = row.get("content") or ""
        if not content.strip():
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

    vectorstore.add_vectors(vectors, metadatas)
    vectorstore.save_index()

    clear_embeddings_cache()
    gc.collect()

    return vectorstore


def answer_without_rag(question: str) -> str:
    llm = get_llm()
    system = "You are a helpful assistant. Answer the question concisely."
    return llm.generate(question, system_message=system, max_tokens=200)


def answer_with_rag(vectorstore: FAISSVectorStore, question: str, top_k: int, max_context_chars: int) -> str:
    embeddings = get_embeddings(force_reload=True)
    query_vec = _l2_normalize(embeddings.embed_text(question))

    hits = vectorstore.search_with_metadata(query_vec, k=top_k)

    clear_embeddings_cache()
    gc.collect()

    context_parts: List[str] = []
    total = 0
    for hit in hits:
        md = hit.get("metadata") or {}
        content = (md.get("content") or "").strip()
        if not content:
            continue

        src = md.get("source_file") or "unknown"
        part = f"Source: {src}\nScore: {hit.get('score'):.4f}\n\n{content}"
        if total + len(part) > max_context_chars:
            break

        context_parts.append(part)
        total += len(part)

    context = "\n\n---\n\n".join(context_parts)

    llm = get_llm()
    system = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the context does not contain the answer, say you don't know."
    )
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    return llm.generate(prompt, system_message=system, max_tokens=220)


def write_report(
    out_path: Path,
    question: str,
    without_rag: str,
    with_rag: str,
    chunks_jsonl: Path,
    index_path: Path,
    stats: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"# Q&A Run\n")
    lines.append(f"- Generated at (UTC): `{_utc_now_iso()}`")
    lines.append(f"- Question: **{question}**")
    lines.append(f"- Chunks source: `{chunks_jsonl}`")
    lines.append(f"- FAISS index: `{index_path}`")
    if stats:
        lines.append(f"- Indexed chunks: `{stats.get('indexed_chunks')}`")
        lines.append(f"- Retrieved top_k: `{stats.get('top_k')}`")
    lines.append("\n---\n")

    lines.append("## Answer (No RAG)\n")
    lines.append(without_rag.strip() or "(empty)")
    lines.append("\n---\n")

    lines.append("## Answer (With RAG)\n")
    lines.append(with_rag.strip() or "(empty)")
    lines.append("\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask TinyLlama without RAG vs with RAG")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--out", default="reports/qa_results.md", help="Markdown output path")

    parser.add_argument("--chunks", default="data/chunks/sample_docs.chunks.jsonl", help="Precomputed chunks JSONL")
    parser.add_argument("--index", default=None, help="FAISS index path (defaults to local config)")

    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild FAISS index from chunks")
    parser.add_argument("--max-chunks", type=int, default=800, help="Max chunks to embed/index (for speed)")

    parser.add_argument("--top-k", type=int, default=5, help="How many chunks to retrieve")
    parser.add_argument("--max-context-chars", type=int, default=6000, help="Max retrieved context size")

    args = parser.parse_args()

    os.environ.setdefault("RAG_MODE", "local")

    cfg = get_local_config()
    chunks_path = PROJECT_ROOT / args.chunks

    if not chunks_path.exists():
        raise SystemExit(
            f"Chunks file not found: {chunks_path}. Run: python scripts/precompute_chunks.py"
        )

    index_path = Path(args.index) if args.index else Path(cfg.faiss_index_path)
    if index_path.suffix != ".faiss":
        index_path = index_path.with_suffix(".faiss")

    # Build or reuse index (embeddings step)
    vectorstore = build_index_from_chunks(
        chunks_jsonl=chunks_path,
        index_path=index_path,
        rebuild=args.rebuild_index,
        max_chunks=args.max_chunks,
    )

    indexed_chunks = vectorstore.get_vector_count()

    # Ask without RAG
    without_rag = answer_without_rag(args.question)

    # Ask with RAG
    with_rag = answer_with_rag(
        vectorstore,
        args.question,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
    )

    write_report(
        out_path=PROJECT_ROOT / args.out,
        question=args.question,
        without_rag=without_rag,
        with_rag=with_rag,
        chunks_jsonl=chunks_path,
        index_path=index_path,
        stats={
            "indexed_chunks": indexed_chunks,
            "top_k": args.top_k,
        },
    )

    print(f"✅ Wrote report: {args.out}")


if __name__ == "__main__":
    main()
