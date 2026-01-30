#!/usr/bin/env python3
"""Precompute document chunks to disk.

This script parses all supported files under a directory (default: sample_docs),
chunks them using the existing DocumentProcessor, and writes results to JSONL.

Why JSONL?
- Append-friendly
- Easy to stream later when embedding/indexing
- Keeps one chunk per line

Outputs:
- ./data/chunks/sample_docs.chunks.jsonl
- ./data/chunks/sample_docs.stats.json

Usage:
    python scripts/precompute_chunks.py
    python scripts/precompute_chunks.py --docs src/database/sample_docs --out data/chunks/sample_docs.chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is importable so `import src.*` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.document_processor import DocumentProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute chunks for a document folder")
    parser.add_argument("--docs", default="src/database/sample_docs", help="Directory containing documents")
    parser.add_argument("--out", default="data/chunks/sample_docs.chunks.jsonl", help="Output JSONL path")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")

    args = parser.parse_args()

    docs_dir = Path(args.docs)
    out_path = Path(args.out)
    stats_path = out_path.with_suffix(".stats.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    processor = DocumentProcessor(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = processor.process_directory(str(docs_dir))

    # Write JSONL
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            row = {
                "content": chunk.content,
                "metadata": chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "file_type": chunk.file_type,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = processor.get_processing_stats(chunks)
    stats["docs_dir"] = str(docs_dir)
    stats["output"] = str(out_path)
    stats["chunk_size"] = args.chunk_size
    stats["chunk_overlap"] = args.chunk_overlap
    stats["generated_at"] = datetime.now(timezone.utc).isoformat()

    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"✅ Wrote chunks: {out_path} ({stats['total_chunks']} chunks)")
    print(f"✅ Wrote stats:  {stats_path}")


if __name__ == "__main__":
    main()
