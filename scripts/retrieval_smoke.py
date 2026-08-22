"""Run the D1 ten-query retrieval smoke and write a reviewable Markdown report."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import DashScopeEmbeddings

from ingest import DEFAULT_COLLECTION, DEFAULT_DB_DIR, PROJECT_ROOT, load_dotenv_if_present


DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "retrieval_smoke.md"
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "ingest_manifest.json"

SMOKE_CASES = [
    ("RS-001", "容器", "自动续费扣款失败后，系统还会重复尝试吗？", "data/kb/KB-01-容器.md"),
    ("RS-002", "告警", "监控指标需要连续几个周期超阈值才会触发告警？", "data/kb/KB-02-告警.md"),
    ("RS-003", "返佣", "客户订单低于七折时为什么没有返佣？", "data/kb/KB-03-返佣.md"),
    ("RS-004", "寄售", "寄售商品退订时退款金额怎么计算？", "data/kb/KB-04-寄售.md"),
    ("RS-005", "开票", "产生欠票后还能继续申请新发票吗？", "data/kb/KB-05-开票.md"),
    ("RS-006", "算力券", "单个企业最多能申领多少算力券？", "data/kb/KB-06-算力券.md"),
    ("RS-007", "虚拟机", "虚拟机快照如何恢复数据？", "data/kb/KB-07-虚拟机.md"),
    ("RS-008", "订单与交付", "订单为什么一直显示资源分配中？", "data/kb/KB-08-订单与交付.md"),
    ("RS-009", "容器", "公网 IP 绑定容器需要满足哪些状态和资源中心条件？", "data/kb/KB-01-容器.md"),
    ("RS-010", "开票", "退订退款后，已经开过的发票应该怎样处理？", "data/kb/KB-05-开票.md"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "5")))
    return parser.parse_args()


def compact_excerpt(text: str, limit: int = 180) -> str:
    compact = " ".join(text.split()).replace("|", "\\|")
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def load_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    load_dotenv_if_present(PROJECT_ROOT / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[smoke] DASHSCOPE_API_KEY is missing.", file=sys.stderr)
        return 2
    if args.top_k <= 0:
        print("[smoke] top_k must be positive.", file=sys.stderr)
        return 2

    embedding_client = DashScopeEmbeddings(
        model=args.embedding_model,
        dashscope_api_key=api_key,
        max_retries=5,
    )
    client = chromadb.PersistentClient(
        path=str(args.persist_dir.resolve()),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(args.collection)
    except ValueError:
        print("[smoke] Chroma collection does not exist; run scripts/ingest.py first.", file=sys.stderr)
        return 2
    collection_count = collection.count()
    if collection_count == 0:
        print("[smoke] Chroma collection is empty; run scripts/ingest.py first.", file=sys.stderr)
        return 2

    manifest = load_manifest(args.manifest)
    results: list[dict[str, object]] = []
    latencies_ms: list[float] = []

    for case_id, module, question, expected_source in SMOKE_CASES:
        started_at = time.perf_counter()
        query_vector = embedding_client.embed_query(question)
        query_result = collection.query(
            query_embeddings=[query_vector],
            n_results=args.top_k,
            include=["documents", "metadatas", "distances"],
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        latencies_ms.append(latency_ms)
        retrieved = list(
            zip(
                query_result["documents"][0],
                query_result["metadatas"][0],
                query_result["distances"][0],
                strict=True,
            )
        )
        source_hit = any(metadata.get("source") == expected_source for _, metadata, _ in retrieved)
        results.append(
            {
                "id": case_id,
                "module": module,
                "question": question,
                "expected_source": expected_source,
                "source_hit": source_hit,
                "latency_ms": latency_ms,
                "retrieved": retrieved,
            }
        )
        print(f"[smoke] {case_id} source_hit@{args.top_k}={source_hit} latency_ms={latency_ms:.1f}")

    source_hits = sum(bool(item["source_hit"]) for item in results)
    lines = [
        "# D1 检索 Smoke 报告（BASELINE-001）",
        "",
        f"> 运行时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "> 自动 `source_hit@5` 只检查预期 KB 文件是否进入 top5；最终相关性必须由人工阅读 chunk 后判定。",
        "",
        "## 配置与汇总",
        "",
        "| 项目 | 实测值 |",
        "|---|---:|",
        f"| 入库文档数 | {manifest.get('document_count', '未知')} |",
        f"| 分块数 / collection count | {manifest.get('chunk_count', '未知')} / {collection_count} |",
        f"| embedding 失败数 | {manifest.get('embedding_failure_count', '未知')} |",
        f"| chunk_size / overlap / top_k | {manifest.get('chunk_size', '未知')} / {manifest.get('chunk_overlap', '未知')} / {args.top_k} |",
        f"| embedding model | {args.embedding_model} |",
        f"| 自动 source_hit@{args.top_k} | {source_hits}/{len(results)} |",
        f"| 单次检索平均耗时 | {statistics.fmean(latencies_ms):.1f} ms |",
        f"| 单次检索最小 / 最大耗时 | {min(latencies_ms):.1f} / {max(latencies_ms):.1f} ms |",
        "",
        "## 逐条结果（等待人工判定）",
        "",
    ]

    for item in results:
        lines.extend(
            [
                f"### {item['id']}｜{item['module']}",
                "",
                f"- 问题：{item['question']}",
                f"- 预期来源：`{item['expected_source']}`",
                f"- 自动 source_hit@{args.top_k}：{'通过' if item['source_hit'] else '未通过'}",
                f"- 检索耗时：{item['latency_ms']:.1f} ms",
                "- 人工判定：`[ ] 相关 / [ ] 不相关`",
                "- 人工备注：`待填写`",
                "",
                "| rank | source | chunk | distance | 内容摘要 |",
                "|---:|---|---:|---:|---|",
            ]
        )
        for rank, (document, metadata, score) in enumerate(item["retrieved"], start=1):
            lines.append(
                "| {rank} | `{source}` | {chunk} | {score:.6f} | {excerpt} |".format(
                    rank=rank,
                    source=metadata.get("source", "未知"),
                    chunk=metadata.get("chunk_index", "未知"),
                    score=score,
                    excerpt=compact_excerpt(document),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## 人工验收汇总",
            "",
            "- 人工判定 top5 相关：`____ / 10`（验收参考：≥8/10）",
            "- 不相关用例及原因：`待填写`",
            "- D1 检索 smoke 最终结论：`[ ] 通过 / [ ] 未通过`",
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[smoke] report={args.report}")
    print(f"[smoke] source_hit@{args.top_k}={source_hits}/{len(results)} avg_latency_ms={statistics.fmean(latencies_ms):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
