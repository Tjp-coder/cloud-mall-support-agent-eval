"""Command-line interface for the D2 RAG customer-support agent."""

from __future__ import annotations

import argparse
import json
import sys

from .rag_agent import RAGAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="Ask one question and exit; omit for interactive mode.")
    parser.add_argument("--json", action="store_true", help="Print the complete response as JSON.")
    return parser.parse_args()


def print_response(question: str, agent: RAGAgent, as_json: bool) -> None:
    response = agent.ask(question)
    if as_json:
        print(json.dumps(response.to_trace(question), ensure_ascii=False, indent=2))
        return
    print(f"\n回答：{response.answer}")
    print("\n检索来源：")
    for source in response.sources:
        print(f"- {source}")
    print(
        f"\n耗时：{response.latency_ms:.1f} ms "
        f"（检索 {response.retrieval_latency_ms:.1f} / 生成 {response.generation_latency_ms:.1f}）"
    )
    print(f"Token：输入 {response.input_tokens} / 输出 {response.output_tokens}")


def main() -> int:
    args = parse_args()
    try:
        agent = RAGAgent.from_env()
        if args.question:
            print_response(args.question, agent, args.json)
            return 0

        print("算力云商城客服 RAG CLI；输入 exit 或 quit 退出。")
        while True:
            question = input("\n问题> ").strip()
            if question.lower() in {"exit", "quit"}:
                return 0
            if not question:
                continue
            try:
                print_response(question, agent, args.json)
            except (RuntimeError, ValueError) as exc:
                print(f"调用失败：{exc}", file=sys.stderr)
    except (RuntimeError, ValueError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
