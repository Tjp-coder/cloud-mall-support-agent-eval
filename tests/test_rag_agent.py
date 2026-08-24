"""Offline unit tests for the D2 RAG chain."""

from __future__ import annotations

import unittest

from agent.rag_agent import Completion, RAGAgent, RetrievedChunk, SYSTEM_PROMPT


class FakeEmbeddingClient:
    def embed_query(self, text: str) -> list[float]:
        self.last_text = text
        return [0.1, 0.2]


class FakeCollection:
    def query(self, **kwargs: object) -> dict[str, object]:
        self.last_kwargs = kwargs
        return {
            "documents": [["规则A", "规则B"]],
            "metadatas": [[
                {"source": "data/kb/KB-01-容器.md", "chunk_index": 5},
                {"source": "data/kb/KB-01-容器.md", "chunk_index": 6},
            ]],
            "distances": [[0.2, 0.3]],
        }


class FakeCompletionClient:
    def complete(self, messages: list[dict[str, str]]) -> Completion:
        self.last_messages = messages
        return Completion("依据上下文回答。", 123, 17, "fake-qwen")


class RAGAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding = FakeEmbeddingClient()
        self.collection = FakeCollection()
        self.completion = FakeCompletionClient()
        self.agent = RAGAgent(
            embedding_client=self.embedding,
            collection=self.collection,
            completion_client=self.completion,
            top_k=2,
        )

    def test_ask_retrieves_generates_and_returns_trace_fields(self) -> None:
        response = self.agent.ask("自动续费失败会重试吗？")

        self.assertEqual(response.answer, "依据上下文回答。")
        self.assertEqual(response.sources, ["data/kb/KB-01-容器.md"])
        self.assertEqual([item.rank for item in response.contexts], [1, 2])
        self.assertEqual(response.input_tokens, 123)
        self.assertEqual(response.output_tokens, 17)
        self.assertGreaterEqual(response.latency_ms, 0)
        trace = response.to_trace("自动续费失败会重试吗？", case_id="TEST-001")
        self.assertEqual(trace["case_id"], "TEST-001")
        self.assertEqual(len(trace["contexts"]), 2)

    def test_prompt_contains_context_sources_and_safety_rules(self) -> None:
        contexts = [RetrievedChunk(1, "data/kb/KB-03-返佣.md", 3, 0.2, "低于七折无返佣")]
        messages = RAGAgent.build_messages("为什么没有返佣？", contexts)

        self.assertIn("只能依据", SYSTEM_PROMPT)
        self.assertIn("条件性规则", SYSTEM_PROMPT)
        self.assertIn("data/kb/KB-03-返佣.md", messages[1]["content"])
        self.assertIn("低于七折无返佣", messages[1]["content"])

    def test_empty_question_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.agent.ask("  ")


if __name__ == "__main__":
    unittest.main()
