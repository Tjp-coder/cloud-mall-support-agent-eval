# 算力云商城客服问答 Agent 测试专项

> 这是我基于自己之前在算力云商城业务上的理解，自己整理编写的一套模拟业务文档，做的一个 Agent 测试方法论实践项目。
> 项目定位:**Agent 只是被测系统(SUT),真正的产出是测试方案、测试集、评测框架、缺陷单、对比报告。**

## 当前进度与 D0 回顾

- 当前日期：2026-08-22，正式开始 D1；D1–D15 执行窗口为 2026-08-22 至 2026-09-05。
- D0 已于 2026-08-16 完成：建立测试范围、风险、策略、指标与排期，整理 8 个模块的模拟 KB，形成 50 条候选 FAQ，初始化项目目录与自检清单。
- 2026-08-17 至 2026-08-21 因面试准备暂停，项目如实顺延，不将暂停期记为已完成工时。

## 项目结构

```
├── agent/              # RAG 问答链、prompt、CLI(D2 起)
├── app/                # FastAPI /chat /health(D3 起)
├── mock_server/        # 3 个 mock 工具 + 故障注入(D3 起)
├── data/kb/            # 8 份模拟业务文档(KB-01~KB-08,已冻结)
├── chroma_db/          # 向量库持久化(D1 起)
├── scripts/            # ingest.py 清洗+分块+embedding+入库(D1)
├── eval/               # dataset_v1.jsonl / calibration_30.jsonl / harness/ / traces/ / reports/
├── tests/              # pytest:smoke / tool / fault / schema
├── docs/               # TEST_PLAN.md / TEST_DESIGN.md / TRACEABILITY_MATRIX.md / final_report.md / INTERVIEW_QA_MAP.md
├── doc/                # 工作区规划文档(完整计划/自检清单/模板/核对清单/审核表)
└── DEFECTS.md          # 缺陷单
```

## 快速开始（D1 基线）

```powershell
# 1. 使用 Python 3.11 创建环境并安装锁定依赖
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. 复制本地配置并填写 DASHSCOPE_API_KEY（.env 不会进入 Git）
Copy-Item .env.example .env

# 3. 重建 BASELINE-001 向量库并执行 10 条检索 smoke
.\.venv\Scripts\python.exe scripts\ingest.py --rebuild
.\.venv\Scripts\python.exe scripts\retrieval_smoke.py
```

> Windows 基线环境固定为 Python 3.11。当前机器上的 Python 3.14 + Chroma 1.x 曾稳定复现原生层 access violation，因此未用于本项目基线。

## D1 当前实测

- 入库：8 篇 KB、84 个 chunk、embedding 失败 0，collection count 84。
- 配置：`text-embedding-v3`、`chunk_size=600`、`overlap=100`、`top_k=5`。
- 自动 `source_hit@5`：10/10；平均耗时 194.0 ms（包含 query embedding 与本地 Chroma 检索）。
- 逐条阅读完整 chunk 后，Top 5 语义相关 10/10（Codex 辅助复核）；RS-006 排序和 RS-008 内容覆盖记为 D7 观察项，D1 未临时调参。

## 关键资产

- 8 份 KB 模拟文档:`data/kb/`（约 313 个唯一规则编号）
- 测试方案:`docs/TEST_PLAN.md`;候选 FAQ:`docs/FAQ_CANDIDATES.md`
- D1 入库清单与检索报告：`outputs/ingest_manifest.json`、`outputs/retrieval_smoke.md`
- 面试自检:`doc/算力云商城Agent测试-自检计划清单.md`(10 题,面试前唯一复习文件)
