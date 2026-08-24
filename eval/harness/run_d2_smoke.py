"""Run the D2 ten-case RAG answer smoke and produce traces plus a review report."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.rag_agent import RAGAgent  # noqa: E402


DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "smoke_v0.md"
DEFAULT_TRACE_DIR = PROJECT_ROOT / "eval" / "traces"

SMOKE_CASES = [
    {
        "id": "D2-001",
        "module": "容器",
        "question": "自动续费扣款失败后，系统还会再次尝试吗？",
        "expected_action": "answer",
        "expected_source": "data/kb/KB-01-容器.md",
    },
    {
        "id": "D2-002",
        "module": "告警",
        "question": "告警默认需要连续几个采集周期达到阈值才会触发？",
        "expected_action": "answer",
        "expected_source": "data/kb/KB-02-告警.md",
    },
    {
        "id": "D2-003",
        "module": "返佣",
        "question": "按照知识库中的参考配置，客户订单实际折扣低于七折时为什么没有返佣？",
        "expected_action": "answer",
        "expected_source": "data/kb/KB-03-返佣.md",
    },
    {
        "id": "D2-004",
        "module": "寄售",
        "question": "寄售商品包年包月退订时，退款金额如何计算？",
        "expected_action": "answer",
        "expected_source": "data/kb/KB-04-寄售.md",
    },
    {
        "id": "D2-005",
        "module": "开票",
        "question": "客户存在未处理欠票时，还能申请新发票吗？",
        "expected_action": "answer",
        "expected_source": "data/kb/KB-05-开票.md",
    },
    {
        "id": "D2-006",
        "module": "算力券",
        "question": "同一活动中，单个企业累计最多能申领多少算力券？",
        "expected_action": "answer",
        "expected_source": "data/kb/KB-06-算力券.md",
    },
    {
        "id": "D2-007",
        "module": "虚拟机",
        "question": "虚拟机如何通过快照恢复数据？",
        "expected_action": "answer",
        "expected_source": "data/kb/KB-07-虚拟机.md",
    },
    {
        "id": "D2-008",
        "module": "订单与交付",
        "question": "订单处于资源分配中时系统正在做什么，预占成功后进入什么状态？",
        "expected_action": "answer",
        "expected_source": "data/kb/KB-08-订单与交付.md",
    },
    {
        "id": "D2-009",
        "module": "库外",
        "question": "今天北京的天气怎么样？",
        "expected_action": "refuse",
        "expected_source": None,
    },
    {
        "id": "D2-010",
        "module": "库外",
        "question": "请写一首关于秋天的七言绝句。",
        "expected_action": "refuse",
        "expected_source": None,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--limit", type=int, help="Run only the first N cases for debugging.")
    return parser.parse_args()


def excerpt(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split()).replace("|", "\\|")
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def blockquote(text: str) -> list[str]:
    lines = text.splitlines() or [""]
    return [f"> {line.rstrip()}" if line.rstrip() else ">" for line in lines]


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        print("[d2-smoke] --limit must be positive", file=sys.stderr)
        return 2
    cases = SMOKE_CASES[: args.limit] if args.limit else SMOKE_CASES
    try:
        agent = RAGAgent.from_env()
    except (RuntimeError, ValueError) as exc:
        print(f"[d2-smoke] startup failed: {exc}", file=sys.stderr)
        return 2

    run_at = datetime.now().astimezone()
    run_id = run_at.strftime("%Y%m%dT%H%M%S%z")
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.trace_dir / f"d2_smoke_{run_id}.jsonl"
    results: list[dict[str, object]] = []

    with trace_path.open("w", encoding="utf-8") as trace_file:
        for case in cases:
            print(f"[d2-smoke] running {case['id']} {case['question']}")
            try:
                response = agent.ask(str(case["question"]))
                trace = response.to_trace(
                    str(case["question"]),
                    run_id=run_id,
                    case_id=case["id"],
                    module=case["module"],
                    expected_action=case["expected_action"],
                    expected_source=case["expected_source"],
                    error=None,
                )
            except Exception as exc:  # retain evidence for external API failures
                trace = {
                    "run_id": run_id,
                    "case_id": case["id"],
                    "module": case["module"],
                    "question": case["question"],
                    "expected_action": case["expected_action"],
                    "expected_source": case["expected_source"],
                    "answer": "",
                    "sources": [],
                    "contexts": [],
                    "latency_ms": 0,
                    "retrieval_latency_ms": 0,
                    "generation_latency_ms": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "model": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
            trace_file.flush()
            results.append(trace)

    successful = [result for result in results if not result["error"]]
    latencies = [float(result["latency_ms"]) for result in successful]
    input_tokens = sum(int(result["input_tokens"]) for result in successful)
    output_tokens = sum(int(result["output_tokens"]) for result in successful)
    lines = [
        "# D2 RAG 问答 Smoke 报告（v0）",
        "",
        f"> 运行时间：{run_at.isoformat(timespec='seconds')}",
        f"> Trace：`{trace_path.relative_to(PROJECT_ROOT).as_posix()}`",
        "> 自动执行只负责留存答案、上下文和埋点；可接受性与拒答正确性必须逐条人工判定。",
        "",
        "## 运行汇总",
        "",
        f"- 执行成功：`{len(successful)}/{len(results)}`",
        f"- 平均端到端耗时：`{statistics.fmean(latencies):.1f} ms`" if latencies else "- 平均端到端耗时：`无成功样本`",
        f"- Token 合计：输入 `{input_tokens}` / 输出 `{output_tokens}`",
        "- 人工可接受答案：`____ / 10`（参考：≥8/10）",
        "- 两条库外拒答：`____ / 2`",
        "",
        "## 逐条结果",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"### {result['case_id']}｜{result['module']}｜期望 `{result['expected_action']}`",
                "",
                f"- 问题：{result['question']}",
                f"- 期望来源：`{result['expected_source'] or '无（库外拒答）'}`",
                f"- Top 5 检索来源：`{', '.join(result['sources']) if result['sources'] else '无'}`",
                f"- 耗时：总计 `{float(result['latency_ms']):.1f} ms`；检索 `{float(result['retrieval_latency_ms']):.1f} ms`；生成 `{float(result['generation_latency_ms']):.1f} ms`",
                f"- Token：输入 `{result['input_tokens']}` / 输出 `{result['output_tokens']}`",
                f"- 执行错误：`{result['error'] or '无'}`",
                "- 模型回答：",
                "",
                *blockquote(str(result["answer"]) or "（无回答）"),
                "",
                "- Top 5 上下文摘要：",
                "",
            ]
        )
        for context in result["contexts"]:
            lines.append(
                "  - rank {rank}｜`{source}` chunk {chunk_index}｜distance {distance:.6f}｜{content}".format(
                    rank=context["rank"],
                    source=context["source"],
                    chunk_index=context["chunk_index"],
                    distance=float(context["distance"]),
                    content=excerpt(str(context["content"])),
                )
            )
        lines.extend(
            [
                "",
                "- 人工判定：`[ ] 可接受 / [ ] 不可接受`",
                "- 事实与条件：`待填写`",
                "- 拒答/澄清行为：`待填写`",
                "- 引用来源：`待填写`",
                "- 备注：`待填写`",
                "",
            ]
        )

    lines.extend(
        [
            "## 人工验收结论",
            "",
            "- 可接受答案：`____ / 10`",
            "- 库外拒答：`____ / 2`",
            "- Trace 字段完整：`____ / 10`",
            "- 失败用例及归因：`待填写`",
            "- D2 最终结论：`[ ] 通过 / [ ] 未通过`",
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[d2-smoke] trace={trace_path}")
    print(f"[d2-smoke] report={args.report}")
    print(f"[d2-smoke] successful={len(successful)}/{len(results)}")
    return 0 if len(successful) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
