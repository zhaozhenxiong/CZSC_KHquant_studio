---
type: operation_note
updated_at: 2026-10-09
audit_session: 20261008-233054-6938018c
---

# 数据质量报告保存

`DataQualityChecker.save_report()` 原来引用已经移除的 `DataAccessLayer.processed`，导致行情更新完成后的质量报告无法保存。现改用当前 `storage.raw.transaction()`，把 warning、error 等诊断记录写入原始行情库的既有 `data_quality_issues` 表；不创建 processed warehouse，不修改行情。

每条问题的 `issue_id` 包含报告 run_id、股票、分类和消息哈希，`source_file` 保存报告 run_id。同一报告重复保存采用 upsert；任意记录不符合数据库约束时，整批事务回滚。

完整报告使用现有 RunContext 保存：

- `artifacts/runs/<report_run_id>/validation/data_quality_issues.csv`：问题列表；无问题时仍保留列头。
- `artifacts/runs/<report_run_id>/validation/data_quality_report.json`：检查区间、股票、逐股统计、摘要和问题。
- 同目录上层的 `metadata.json`：保存状态、原始数据库路径和报告路径。

行情更新入口的报告 run_id 是 `<update_id>_quality`。文件存储模式只保存这些 artifacts，不创建 SQLite 数据库。质量检查发现问题和报告保存成功是两个独立结果；metadata 的 `completed` 表示报告已保存，`data_quality.has_errors` 表示是否存在 error。

修复保持原有诊断逻辑及独立研究执行 gate 的行为。诊断检查的行情日期并集不成为已核验研究交易日历，warning 不被改写为 error。本轮最初全市场更新所用旧代码不会被追溯修改，需要在修复后的重试或独立检查中重新保存。

验证：`PYTHONPATH=./app .venv/bin/python -m pytest -q app/my_strategy/tests/test_data_quality.py app/my_strategy/tests/test_data_quality_gate.py`，14 项通过。隔离测试实际写入临时 raw SQLite，覆盖 warning/error 持久化、重复保存幂等、约束错误回滚、完整 JSON/CSV 和文件模式无数据库副作用。
