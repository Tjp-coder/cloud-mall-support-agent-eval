"""Build the D1 Chroma retrieval baseline from the simulated KB documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB_DIR = PROJECT_ROOT / "data" / "kb"
DEFAULT_DB_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "ingest_manifest.json"
DEFAULT_COLLECTION = "cloud_mall_support_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"))
    parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "600")))
    parser.add_argument("--overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "100")))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate only the configured Chroma collection before ingesting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and split the KB without calling DashScope or writing Chroma.",
    )
    return parser.parse_args()


def load_dotenv_if_present(path: Path) -> None:
    """Load simple KEY=VALUE entries without adding a runtime dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_kb_documents(kb_dir: Path) -> list[Document]:
    files = sorted(kb_dir.glob("KB-*.md"))
    if not files:
        raise FileNotFoundError(f"No KB markdown files found under: {kb_dir}")

    documents: list[Document] = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"KB file is empty: {path}")
        relative_source = path.relative_to(PROJECT_ROOT).as_posix()
        kb_id = path.stem.split("-", 2)[:2]
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": relative_source,
                    "filename": path.name,
                    "kb_id": "-".join(kb_id),
                    "document_sha256": file_sha256(path),
                },
            )
        )
    return documents


def split_documents(documents: list[Document], chunk_size: int, overlap: int) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks: list[Document] = []
    for document in documents:
        document_chunks = splitter.split_documents([document])
        for index, chunk in enumerate(document_chunks):
            stable_material = f"{chunk.metadata['source']}:{index}:{chunk.page_content}"
            chunk_id = hashlib.sha256(stable_material.encode("utf-8")).hexdigest()
            chunk.metadata.update({"chunk_index": index, "chunk_id": chunk_id})
            chunks.append(chunk)
    return chunks


def build_manifest(
    *,
    args: argparse.Namespace,
    documents: list[Document],
    chunks: list[Document],
    embedding_failures: int,
    elapsed_seconds: float,
    collection_count: int | None,
) -> dict[str, object]:
    chunks_by_source = Counter(str(chunk.metadata["source"]) for chunk in chunks)
    return {
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "baseline_name": "BASELINE-001",
        "dry_run": args.dry_run,
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.overlap,
        "batch_size": args.batch_size,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "collection_count": collection_count,
        "embedding_failure_count": embedding_failures,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "chunks_by_source": dict(sorted(chunks_by_source.items())),
    }


def main() -> int:
    args = parse_args()
    load_dotenv_if_present(PROJECT_ROOT / ".env")
    started_at = time.perf_counter()

    try:
        documents = load_kb_documents(args.kb_dir.resolve())
        chunks = split_documents(documents, args.chunk_size, args.overlap)
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as exc:
        print(f"[ingest] preparation failed: {exc}", file=sys.stderr)
        return 2

    print(f"[ingest] documents={len(documents)} chunks={len(chunks)}")
    embedding_failures = 0
    collection_count: int | None = None

    if not args.dry_run:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            print(
                "[ingest] DASHSCOPE_API_KEY is missing. Configure it in the process environment or local .env.",
                file=sys.stderr,
            )
            return 2

        embedding_client = DashScopeEmbeddings(
            model=args.embedding_model,
            dashscope_api_key=api_key,
            max_retries=5,
        )
        args.persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(args.persist_dir.resolve()),
            settings=Settings(anonymized_telemetry=False),
        )
        if args.rebuild:
            try:
                client.delete_collection(args.collection)
            except ValueError:
                pass
        collection = client.get_or_create_collection(
            name=args.collection,
            metadata={"hnsw:space": "cosine"},
        )

        for offset in range(0, len(chunks), args.batch_size):
            batch = chunks[offset : offset + args.batch_size]
            ids = [str(chunk.metadata["chunk_id"]) for chunk in batch]
            try:
                vectors = embedding_client.embed_documents([chunk.page_content for chunk in batch])
                collection.upsert(
                    ids=ids,
                    documents=[chunk.page_content for chunk in batch],
                    metadatas=[chunk.metadata for chunk in batch],
                    embeddings=vectors,
                )
                print(f"[ingest] embedded {min(offset + len(batch), len(chunks))}/{len(chunks)}")
            except Exception as exc:  # external SDK errors vary by version
                embedding_failures += len(batch)
                print(
                    f"[ingest] batch {offset // args.batch_size + 1} failed ({len(batch)} chunks): {exc}",
                    file=sys.stderr,
                )
        collection_count = collection.count()

    elapsed_seconds = time.perf_counter() - started_at
    manifest = build_manifest(
        args=args,
        documents=documents,
        chunks=chunks,
        embedding_failures=embedding_failures,
        elapsed_seconds=elapsed_seconds,
        collection_count=collection_count,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ingest] manifest={args.manifest}")
    print(f"[ingest] elapsed_seconds={elapsed_seconds:.3f} failures={embedding_failures}")
    return 0 if embedding_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
