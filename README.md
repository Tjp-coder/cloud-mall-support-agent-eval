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

## 快速开始(占位,D1 起逐步填充)

```bash
# 1. 安装依赖 + 配置 .env(DASHSCOPE_API_KEY 等)
# 2. 入库:  python scripts/ingest.py
# 3. 起服务: uvicorn app.main:app
# 4. 评测:  python eval/harness/run.py
# 5. 出报告: 见 eval/reports/
```

## 关键资产

- 8 份 KB 模拟文档:`data/kb/`（约 313 个唯一规则编号）
- 测试方案:`docs/TEST_PLAN.md`;候选 FAQ:`docs/FAQ_CANDIDATES.md`
- 面试自检:`doc/算力云商城Agent测试-自检计划清单.md`(10 题,面试前唯一复习文件)
