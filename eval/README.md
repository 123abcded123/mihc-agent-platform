# Harness 评测模块

对齐《项目文档》5.8 与简历"效果评测"：围绕 HitRate、MRR、Recall 优化检索效果，
用 LLM-as-Judge 评估生成质量，形成"采集-分析-优化-验证"闭环。

## 评测集格式（eval/datasets/*.jsonl，一行一个样本）

```json
{"query": "问题", "relevant_ids": ["doc-001-chunk-00"], "evidence": ["证据片段", ...]}
{"query": "危险问题", "is_dangerous": true}
{"query": "问题", "answer": "AI 回答", "evidence": ["参考证据"]}
```

- 检索评估样本：`query + relevant_ids`（人工标注相关 chunk_id）
- 安全评估样本：`query + is_dangerous`
- 生成评估样本：`query + answer + evidence`（离线历史回答 + 证据）

## 运行

```bash
# 全部评估（检索 + 生成 + 安全）
python eval/run_eval.py

# 只跑检索指标
python eval/run_eval.py --only retrieval

# 保存结果到 MongoDB（无服务时自动落 data/mongo_json/eval_results.json）
python eval/run_eval.py --save
```

## 指标

| 指标 | 说明 |
|---|---|
| HitRate@K | 前 K 个召回中命中相关片段的问题比例 |
| Recall@K | 命中相关片段占全部相关片段的比例 |
| MRR | 第一个相关片段排名的倒数均值 |
| faithfulness | 回答被证据支持程度（LLM-as-Judge） |
| completeness | 覆盖问题关键点程度 |
| citation | 引用正确率 |
| safety_rejection_rate | 危险问题正确拦截比例 |

## 评测集构造

1. 从审计日志/会话记录收集真实问题；
2. 专家标注相关 chunk（或检索后人工确认）；
3. 危险问题从安全事件/人工构造中抽取；
4. 版本化管理（eval/datasets/ 按日期归档），训练集与评测集严格隔离（文档 5.6）。
