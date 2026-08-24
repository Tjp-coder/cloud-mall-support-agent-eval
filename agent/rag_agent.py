"""D2 RAG question-answering chain: retrieve, prompt, generate, and trace."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import DashScopeEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION = "cloud_mall_support_baseline"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
REFUSAL_TEXT = "抱歉，现有知识库没有足够依据回答该问题。"

SYSTEM_PROMPT = f"""你是算力云商城客服问答助手。请严格遵守以下规则：
1. 只能依据用户消息中提供的“知识库上下文”回答，不得使用外部知识或自行补全事实。
2. 上下文与问题无直接关系或没有足够依据时，只回答：{REFUSAL_TEXT}
3. 问题属于业务范围但缺少必要信息时，说明信息不足，并明确请用户补充什么；不得猜测。
4. “参考配置”“可配置”“以实际规则为准”等条件必须保留，不得把条件性规则说成固定规则。
5. 正常回答应简洁、直接，并在末尾用“依据：文件名（chunk N）”列出实际使用的来源。
6. 不执行修改订单、余额、发票等操作；本阶段只提供知识问答。
"""


class EmbeddingClient(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class CollectionClient(Protocol):
    def query(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    source: str
    chunk_index: int
    distance: float
    content: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Completion:
    answer: str
    input_tokens: int
    output_tokens: int
    model: str


class CompletionClient(Protocol):
    def complete(self, messages: Sequence[dict[str, str]]) -> Completion: ...


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    sources: list[str]
    contexts: list[RetrievedChunk]
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    input_tokens: int
    output_tokens: int
    model: str

    def to_trace(self, question: str, **extra: object) -> dict[str, object]:
        return {
            "question": question,
            "answer": self.answer,
            "sources": self.sources,
            "contexts": [context.to_dict() for context in self.contexts],
            "latency_ms": self.latency_ms,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
            **extra,
        }


def load_dotenv_if_present(path: Path) -> None:
    """Load simple KEY=VALUE entries without printing or overwriting secrets."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def configured_value(name: str, fallback: str | None = None) -> str | None:
    value = os.getenv(name, fallback)
    if not value or value.startswith("your_"):
        return None
    return value


class QwenCompatibleClient:
    """Small OpenAI-compatible chat client using only the standard library."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "qwen-plus",
        temperature: float = 0.1,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def complete(self, messages: Sequence[dict[str, str]]) -> Completion:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": list(messages),
                "temperature": self.temperature,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "cloud-mall-support-agent-eval/BASELINE-001",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Qwen request failed with HTTP {exc.code}: {error_body}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Qwen request failed: {exc}") from exc

        try:
            answer = str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Qwen response is missing choices[0].message.content") from exc
        usage = body.get("usage") or {}
        return Completion(
            answer=answer,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            model=str(body.get("model") or self.model),
        )


class RAGAgent:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        collection: CollectionClient,
        completion_client: CompletionClient,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.embedding_client = embedding_client
        self.collection = collection
        self.completion_client = completion_client
        self.top_k = top_k

    @classmethod
    def from_env(
        cls,
        *,
        persist_dir: Path = DEFAULT_DB_DIR,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> "RAGAgent":
        load_dotenv_if_present(PROJECT_ROOT / ".env")
        embedding_key = configured_value("DASHSCOPE_API_KEY")
        if not embedding_key:
            raise RuntimeError("DASHSCOPE_API_KEY is missing; copy .env.example to .env and configure it.")
        qwen_key = configured_value("QWEN_API_KEY") or embedding_key

        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
        top_k = int(os.getenv("TOP_K", "5"))
        temperature = float(os.getenv("TEMPERATURE", "0.1"))
        embedding_client = DashScopeEmbeddings(
            model=embedding_model,
            dashscope_api_key=embedding_key,
            max_retries=5,
        )
        chroma_client = chromadb.PersistentClient(
            path=str(persist_dir.resolve()),
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            collection = chroma_client.get_collection(collection_name)
        except ValueError as exc:
            raise RuntimeError("Chroma collection is missing; run scripts/ingest.py --rebuild first.") from exc
        if collection.count() == 0:
            raise RuntimeError("Chroma collection is empty; run scripts/ingest.py --rebuild first.")

        completion_client = QwenCompatibleClient(
            api_key=qwen_key,
            base_url=os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL),
            model=os.getenv("QWEN_MODEL", "qwen-plus"),
            temperature=temperature,
        )
        return cls(
            embedding_client=embedding_client,
            collection=collection,
            completion_client=completion_client,
            top_k=top_k,
        )

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        query_vector = self.embedding_client.embed_query(question)
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            RetrievedChunk(
                rank=rank,
                source=str(metadata.get("source", "未知")),
                chunk_index=int(metadata.get("chunk_index", -1)),
                distance=float(distance),
                content=str(document),
            )
            for rank, (document, metadata, distance) in enumerate(
                zip(documents, metadatas, distances, strict=True),
                start=1,
            )
        ]

    @staticmethod
    def build_messages(question: str, contexts: Sequence[RetrievedChunk]) -> list[dict[str, str]]:
        context_text = "\n\n".join(
            "[上下文 {rank}]\n来源：{source}（chunk {chunk}）\n内容：{content}".format(
                rank=context.rank,
                source=context.source,
                chunk=context.chunk_index,
                content=context.content,
            )
            for context in contexts
        )
        user_message = f"知识库上下文：\n{context_text or '（无）'}\n\n用户问题：{question}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    def ask(self, question: str) -> RAGResponse:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        total_started = time.perf_counter()
        retrieval_started = time.perf_counter()
        contexts = self.retrieve(normalized_question)
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000

        generation_started = time.perf_counter()
        completion = self.completion_client.complete(self.build_messages(normalized_question, contexts))
        generation_latency_ms = (time.perf_counter() - generation_started) * 1000
        latency_ms = (time.perf_counter() - total_started) * 1000
        sources = list(dict.fromkeys(context.source for context in contexts))
        return RAGResponse(
            answer=completion.answer,
            sources=sources,
            contexts=contexts,
            latency_ms=round(latency_ms, 1),
            retrieval_latency_ms=round(retrieval_latency_ms, 1),
            generation_latency_ms=round(generation_latency_ms, 1),
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            model=completion.model,
        )
