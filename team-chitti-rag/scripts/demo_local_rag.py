#!/usr/bin/env python3
"""Demo: Local RAG (TinyLlama + Local Embeddings + FAISS).

Runs an end-to-end flow:
1) Chunk documents
2) Embed chunks
3) Index into FAISS
4) Retrieve top-k chunks for a question
5) Ask TinyLlama using retrieved context

Designed for Codespaces (CPU).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import gc

import numpy as np

# Ensure repo root is importable so `import src.*` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.local_config import get_local_config
from src.database.document_processor import DocumentProcessor
from src.embeddings.embeddings_factory import get_embeddings, clear_embeddings_cache
from src.llm.llm_factory import get_llm
from src.vectorstore.faiss_vectorstore import FAISSVectorStore


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec))
    if denom == 0.0:
        return vec.astype(np.float32)
    return (vec / denom).astype(np.float32)


def build_or_load_index(
    docs_dir: str,
    index_path: str,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int | None,
    rebuild: bool,
) -> FAISSVectorStore:
    embeddings = get_embeddings()

    if not index_path.endswith(".faiss"):
        index_path = index_path + ".faiss"

    vectorstore = FAISSVectorStore({
        "index_path": index_path,
        "index_type": "Flat",  # Best quality for small/medium datasets
        "nlist": 100,
    })

    if (not rebuild) and vectorstore.get_vector_count() > 0:
        return vectorstore

    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = processor.process_directory(docs_dir)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    vectors: List[np.ndarray] = []
    metadatas: List[Dict[str, Any]] = []

    for chunk in chunks:
        vec = embeddings.embed_text(chunk.content)
        vec = _l2_normalize(vec)
        vectors.append(vec)
        metadatas.append({
            "chunk_id": chunk.chunk_id,
            "source_file": chunk.source_file,
            "file_type": chunk.file_type,
            "content": chunk.content,
            "metadata": chunk.metadata,
        })

    vectorstore.add_vectors(vectors, metadatas)
    vectorstore.save_index()

    # Free embeddings model memory before we load the LLM.
    clear_embeddings_cache()
    gc.collect()

    return vectorstore


def answer_question(vectorstore: FAISSVectorStore, question: str, top_k: int, max_context_chars: int) -> str:
    embeddings = get_embeddings()
    query_vec = _l2_normalize(embeddings.embed_text(question))

    # Free embeddings before loading TinyLlama to keep peak memory low.
    clear_embeddings_cache()
    gc.collect()

    llm = get_llm()
    hits = vectorstore.search_with_metadata(query_vec, k=top_k)

    context_parts: List[str] = []
    total = 0
    for hit in hits:
        content = (hit.get("metadata") or {}).get("content") or ""
        if not content:
            continue
        part = f"[score={hit.get('score'):.4f}]\n{content.strip()}"
        if total + len(part) > max_context_chars:
            break
        context_parts.append(part)
        total += len(part)

    context = "\n\n---\n\n".join(context_parts)

    system = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the context does not contain the answer, say you don't know. Keep the answer concise."
    )

    prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

    return llm.generate(prompt, system_message=system, max_tokens=256)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG demo (TinyLlama + FAISS)")
    parser.add_argument("--docs", default="src/database/sample_docs", help="Directory containing documents")
    parser.add_argument("--question", default="What is NEP 2020?", help="Question to ask")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the FAISS index")
    parser.add_argument("--max-chunks", type=int, default=300, help="Max chunks to index (limit for speed)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k chunks to retrieve")
    parser.add_argument("--max-context-chars", type=int, default=6000, help="Max retrieved context length")

    args = parser.parse_args()

    os.environ.setdefault("RAG_MODE", "local")

    cfg = get_local_config()
    top_k = args.top_k or cfg.top_k

    vectorstore = build_or_load_index(
        docs_dir=args.docs,
        index_path=cfg.faiss_index_path,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        max_chunks=args.max_chunks,
        rebuild=args.rebuild,
    )

    print("\n=== Retrieved Answer ===\n")
    answer = answer_question(vectorstore, args.question, top_k=top_k, max_context_chars=args.max_context_chars)
    print(answer)


if __name__ == "__main__":
    main()
