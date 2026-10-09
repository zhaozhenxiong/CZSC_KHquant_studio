# CZSC 工作台

当前唯一分析与策略链路：原始 SQLite 日线 → CZSC BarGenerator/CZSC → Signal/Event/Position → 次日 BrokerSimulator → 成交账本与新结果库。六个网页入口共用同一服务，网页不计算另一套评分。

配置为 `configs/czsc_strategy.json`；行情读取边界为 `adapters/czsc_adapter.py`；结构分析与回测位于 `services/czsc_analysis.py`、`services/czsc_backtest.py`。框架源码位于应用根 `vendor/czsc`，来源校验在 `czsc-source-manifest.json`。

运行 `python -m my_strategy.cli --help`。所有相对业务路径以应用根为基准，使用仓库 `.env` 的 `KHQUANT_DATA_ROOT`、`KHQUANT_RAW_DB`、`KHQUANT_ARTIFACT_ROOT`、`KHQUANT_METADATA_ROOT`、`KHQUANT_LOG_ROOT`。

可视化支持日/周/月切换、分型、笔、中枢、量、截止日回放、净值/回撤和实际成交定位。扫描展示当日已确认事件及失败股票；任务支持进度、重复提交复用、取消和新运行记录。形成中的结构会变化，回放必须重新按当时输入计算。

“我的关注”与扫描星标统一持久保存。“我的持仓”手动管理股票、股数和平均成本，按本地最新收盘价展示市值及浮动盈亏，明确行情日期和缺价记录。数据保存在 `KHQUANT_METADATA_ROOT/personal_portfolio.db`；不与回测持仓混用。六页分别为结构分析、市场扫描、策略回测、我的关注、我的持仓、数据与任务。

旧 V2/Z1/chip/ML/训练/注册/证据链与旧网页已删除，无历史迁移与兼容入口。历史知识库是审计记录，不能作为当前恢复旧模块的指令。

新增研究模式由 `configs/czsc_research.json` 和 `services/czsc_research*.py` 提供：原生已完成笔视图的二/三买卖辅助信号、均线量价、闭合周/月特征，以及 PyTorch MLP 的固定十交易日费用后正收益概率。它是独立研究层；旧训练注册系统继续退役。ML 仅过滤进场，退出使用结构/趋势与原生 Position 风控。

先核验数据及交易日历，再运行 `python -m my_strategy.cli research-train --end 2026-09-30 --calendar-run-id czsc-calendar-baostock-20261001 --device cuda:1 --cpu-workers 8`。随后用 `scan --research --model-run-id <训练运行ID> --end 2026-09-30` 扫描。网页扫描与回测可选择研究模式、模型运行，展示买入候选/观察/退出/排除、证据和影子概率。

模型有三段滚动样本外窗口及冻结留出窗口，训练/验证标签按到期日清除跨界样本，训练段预处理、验证段校准。四种策略使用相同股票池、固定等额独立账户与真实费用账本，失败配额保留现金。只有完整池多窗口改进门槛通过才允许研究买入分类；否则输出观察及排除，不能声称概率已证实更好的买卖点。缺应执行日行情拒单，不用更晚行情替代。结果均在独立 `artifacts/runs/` 留存。
