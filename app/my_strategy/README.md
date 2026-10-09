# CZSC 工作台

当前唯一分析与策略链路：原始 SQLite 日线 → CZSC BarGenerator/CZSC → Signal/Event/Position → 次日 BrokerSimulator → 成交账本与新结果库。六个网页入口共用同一服务，网页不计算另一套评分。

配置为 `configs/czsc_strategy.json`；行情读取边界为 `adapters/czsc_adapter.py`；结构分析与回测位于 `services/czsc_analysis.py`、`services/czsc_backtest.py`。框架源码位于应用根 `vendor/czsc`，来源校验在 `czsc-source-manifest.json`。

运行 `python -m my_strategy.cli --help`。所有相对业务路径以应用根为基准，使用仓库 `.env` 的 `KHQUANT_DATA_ROOT`、`KHQUANT_RAW_DB`、`KHQUANT_ARTIFACT_ROOT`、`KHQUANT_METADATA_ROOT`、`KHQUANT_LOG_ROOT`。

可视化支持日/周/月切换、分型、笔、中枢、量、截止日回放、净值/回撤和实际成交定位。扫描展示当日已确认事件及失败股票；任务支持进度、重复提交复用、取消和新运行记录。形成中的结构会变化，回放必须重新按当时输入计算。

“我的关注”与扫描星标统一持久保存。“我的持仓”手动管理股票、股数和平均成本，按本地最新收盘价展示市值及浮动盈亏，明确行情日期和缺价记录。数据保存在 `KHQUANT_METADATA_ROOT/personal_portfolio.db`；不与回测持仓混用。六页分别为结构分析、市场扫描、策略回测、我的关注、我的持仓、数据与任务。

旧 V2/Z1/chip/ML/训练/注册/证据链与旧网页已删除，无历史迁移与兼容入口。历史知识库是审计记录，不能作为当前恢复旧模块的指令。

研究模式包含均线20项专家、结构22项专家、威科夫量价32项专家，以及严格时序样本外概率融合。均线和结构的 PyTorch MLP、量价及融合的正则线性模型分别保留契约；10日概率目标为下一核验开盘买入、持有10交易日、费用后正收益。逐日事件输出锚点、首次观测与可用时间，图上只在确认日显示。对应配置为 `configs/czsc_research.json`、`configs/czsc_dual_research.json` 和 `configs/czsc_wyckoff_research.json`，旧训练注册系统继续退役。

先核验数据及交易日历，再运行 `python -m my_strategy.cli research-train --end 2026-09-30 --calendar-run-id czsc-calendar-baostock-20261001 --device cuda:1 --cpu-workers 8`。随后用 `scan --research --model-run-id <训练运行ID> --end 2026-09-30` 扫描。网页扫描与回测可选择研究模式、模型运行，展示买入候选/观察/退出/排除、证据和影子概率。

威科夫全量入口为 `python -m my_strategy.cli research-wyckoff-train --source-training-run-id <冻结均线源运行ID> --end 2026-10-08 --device mps --cpu-workers 4 --json`。不传 `--stocks` 时覆盖冻结源全池。分析、扫描和回测使用 `--model-family wyckoff`；历史自动路由逐日切换检查点，普通研究保持规则与连续账本。无量价模型时回到具有同一目标、核验日历和来源的双专家影子概率；无可用模型时继续规则。生产时点还受实际训练完成时间限制，未来认证结果不能回填。

独立退出头仅读取实际 Broker 持仓；独立策略入场头读取真实空仓、候选和资金风险状态。二者各有费用后目标，不用 `1-p_buy` 解释卖出，也不拿固定10日概率替代实际策略概率。普通分析、扫描、回测仅展示影子；模型应用仅发生于明确的离线实验臂。原生退出、实际止损及强制退出优先，个人关注和手动持仓与研究账户独立。

威科夫研究按冻结配置评估四个已观察历史开发窗口、14个策略臂和3个辅助对照，并以真实 Broker 回放检查费用和滑点压力。训练/验证标签按成熟日清除跨界样本，预处理只拟合训练段。固定股票池采用等额独立现金账户，失败账户保留现金，缺应执行日行情拒单。历史 ST、退市池及公司行动身份尚缺，现有目标仍沿用未复权原始价格；数据质量过滤不能替代分红与份额修复。模型保持影子，历史开发结果与工程验收均不能证明胜率或盈利改善。独立未来验证开始于最终冻结之后第一个经核验交易日，实际日期写入新的冻结约定。结果均在独立 `artifacts/runs/` 留存。
