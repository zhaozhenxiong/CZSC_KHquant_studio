# 均线趋势 ML 实施与运行记录

审计会话：`20261009-110834-6355e11c`。状态：实施、100股试跑、完整池训练、独立审计、MPS原样重训及最新网页验证全部完成。模型保持影子资格，未建立正式发布。

此前ML为0行的直接原因是没有与所选特征和研究日期兼容的已训练检查点，模型没有加载；不是已经执行了推理却没有展示。现在默认配置已接通均线趋势模型，并完成实际训练和推理。模型加载、推理行数和MPS工作均依据本次执行记录。

## 已完成

- 默认接通 `ma_trend_v1` 的20个均线趋势输入、MLP训练、逻辑回归对照、版本校验、逐日检查点路由及影子推理。
- 概率目标固定为假设新入场后持有10交易日的费用后正收益。无模型日期继续规则研究；切换检查点保留连续Position、仓位和账本；未来发布资格不能回填。
- 模型输入资格与CZSC规则输入资格分离。模型有效而规则输入失格时保留影子概率；目标日无行情和不支持的板块不显示当前概率。
- 网页依据实际模型行数、批次、加载及设备记录显示计算工作。100股及完整池训练均已执行真实MPS工作。
- 225项针对性测试通过，1项依赖CUDA硬件的测试跳过；后续运行掩码修复的18项相关测试复核通过（与此前199项重复，不额外累计）；随后26项新增缓存身份回归通过，同轮18项重复验证不累计。

## 100股试跑

运行：`20261009-112420-746925-20261008-czsc-research-train-NO_GIT`。冻结样本为seed42的50只沪市和50只深市股票。

100/100准备成功、0失败；239,625根冻结行情，研究数据集82,132行。9个模型（5个MLP检查点、4个逻辑回归对照）实际在MPS训练5,360批。实际推理60,565行、394批、12次模型加载。4个账本窗口未通过改进门槛，完整池资格false；保持影子，没有发布。

独立审计：`artifacts/runs/20261009-112420-746925-20261008-czsc-research-train-NO_GIT/reports/execution_audit.json`，0错误。原样重训运行：`20261009-112817-161541-20261008-czsc-research-reproducibility-NO_GIT`；6个状态张量和18个清单字段一致；20行MPS预测差值0，CPU/MPS最大差值5.96e-08，源产物未修改。

实际网页分析通富微电：369行MPS推理，5个按历史日期切换的模型，影子概率50.74%。该值仅是本次模型输出，不构成资格认证或交易建议。

## 跨类型实际扫描

运行：`20261009-114112-229354-20261008-czsc-research-scan-NO_GIT`。五只股票准备均成功；真实MPS推理1,296行、5批、5次模型加载。

002156.SZ、000001.SZ和600000.SH产生真实影子概率。000016.SZ实际行情结束于2026-09-03，目标日缺失，概率为空；688089.SH因当前板块/输入契约不支持，概率为空。此检查没有替代或补填行情。

精简机器证据：[actual-five-stock-scan.json](../../knowledge_base_system/validation/20261009-110834-6355e11c/actual-five-stock-scan.json)。

## 连续历史回测

运行：`20261009-115114-501433-20261008-czsc-research-backtest-NO_GIT`。600000.SH、000001.SZ、002156.SZ，2025-01-01至2026-10-08，fresh规则，3/3成功。

真实MPS推理1,106行、5批、5次模型加载；1,106个股票交易日保持影子，169个无模型股票交易日继续规则。实际Broker账本20笔成交、425个汇总日期，0拒单。逐笔股数和现金、每日账面权益、下一核验交易日执行以及持久化账本行数检查均通过，0错误。切换日现金延续，无重新初始化。

三只股票在检查点切换日均为空仓；跨模型非零理论Position连续性由独立运行回归测试验证，本次真实股票结果不冒充非零仓位跨点证据。[actual-continuous-backtest.json](../../knowledge_base_system/validation/20261009-110834-6355e11c/actual-continuous-backtest.json)。

## 完整池任务

任务ID：`ff96423d5c994e8bace5695f426b5d19`。

运行ID：`20261009-112943-155001-20261008-czsc-research-train-NO_GIT`。

本地5572只股票全部冻结请求，行情截至2026-10-08。完整准备已完成：5572/5572成功、0失败；11,276,786根冻结行情，研究数据集2,624,396行，当前支持主板3197只。4个冻结窗口和候选生产检查点均已完成。MPS实际训练9个模型、169,160批，执行1,923,744行推理、12,435批、12次模型加载。最终生产候选实际训练2,431,996行、验证157,645行；研究数据集2,624,396行不等同全部参与生产权重拟合。完整池改进门槛false：4窗口、0通过，没有发布。

以下是冻结独立账户的实际Broker账本评估。收益率包含无成交账户保留的现金，ML比较使用同一成本和执行契约；分类指标仅作诊断，不替代账本门槛。

| 窗口 | 规则账户收益率 | ML账户收益率 | ML往返笔数 | MLP AUC | 逻辑回归AUC | 门槛 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2025Q2 | -0.4754% | -0.0234% | 19 | 0.5115 | 0.6090 | 未通过 |
| 2025Q4 | -0.6380% | -0.0011% | 26 | 0.5418 | 0.5411 | 未通过 |
| 2026Q2 | 0.1303% | 0.0414% | 89 | 0.4889 | 0.4808 | 未通过 |
| 2026Q3 | -0.8559% | -0.0012% | 0 | 0.5383 | 0.5701 | 未通过 |

完整池独立审计已通过：5572个冻结文件、11,276,786根行情、5个路由模型，0错误。[full-audit-summary.json](../../knowledge_base_system/validation/20261009-110834-6355e11c/full-audit-summary.json)。原样重训运行：`20261009-130830-087567-20261008-czsc-research-reproducibility-NO_GIT`。在相同实际MPS设备上重新训练生产候选23,760批，6/6状态张量、18/18清单字段完全一致；20行MPS预测最大差值0，CPU/MPS最大差值`5.960464477539063e-08`，低于1e-6容差。源数据、模型和研究报告哈希未变。[full-repro-summary.json](../../knowledge_base_system/validation/20261009-110834-6355e11c/full-repro-summary.json)。

MLP结构为20输入、32/16隐藏层，seed42、20轮、batch2048、学习率0.001；概率校准仅使用验证集。固定逻辑回归对照采用相同时间划分和标签，不依据测试结果调门槛。数据上可用日与本次实际训练完成日分别记录；模型实际在2026-10-09训练，历史检查点重建仍保持影子，不能倒填正式认证。

## 最新网页与服务

服务持续运行：[http://127.0.0.1:8124/](http://127.0.0.1:8124/)，PID `36735`；健康检查ok，`active_release`为空。日志：`data/logs/dashboard-20261009-ma-trend.log`。

最新完整池模型分析002156.SZ，研究区间2026-01-01至2026-10-08：实际MPS推理369行、5批、5次模型加载，CPU1个准备进程、缓存命中1次；结构准备0.18秒、推理0.26秒、决策0.13秒。页面显示最终完整池生产候选的真实影子概率49.6%，目标是10交易日费用后正收益；55%为冻结参考门槛。本次未将ML用于入场，规则等待/退出及模型资格分别呈现。固定10日模型目标与fresh实际退出契约不同，继续显示影子。浏览器没有错误或警告日志。

截图：[计算记录](../../artifacts/runs/20261009-112943-155001-20261008-czsc-research-train-NO_GIT/validation/ma-full-mps-ui.png)、[影子概率与资格](../../artifacts/runs/20261009-112943-155001-20261008-czsc-research-train-NO_GIT/validation/ma-full-shadow-probability-ui.png)。

## 缓存身份补充修复

共享runtime接受MA缓存前校验当前profile、version、schema hash、完整有序特征身份、记录与frame一致性、当前质量窗口及60活跃观察契约。语义不同或资格字段缺失则重新计算；物理文件SHA不符仍拒绝。保留legacy旧调用的校验行为。修复不改变完整池 `_prepare` 与训练代码，本次全池从原始行情重建，未复用旧MA缓存。

26项新回归覆盖有效命中、旧身份/乱序/记录矛盾、旧资格窗口、缺失字段、原始行情修订及legacy兼容。受影响的18项既有runtime/扫描测试再次通过；独立累计225通过、1 CUDA硬件测试跳过。

## 命令与证据

```sh
.venv/bin/python app/my_strategy/cli.py research-train --end 2026-10-08 --device mps --cpu-workers 8 --json
.venv/bin/python app/my_strategy/scripts/audit_czsc_research.py --training-run-id 20261009-112943-155001-20261008-czsc-research-train-NO_GIT --workers 8
.venv/bin/python app/my_strategy/scripts/verify_czsc_research_reproducibility.py --training-run-id 20261009-112943-155001-20261008-czsc-research-train-NO_GIT --device mps
```

实际完整池任务由网页任务API提交，参数与上方CLI等效，明确使用已核验日历 `20261008-235742-158531-20261008-czsc-verified-calendar-NO_GIT`。CPU负责结构、量价、标签和账本；MPS负责实际模型训练/推理。

再次启动网页可在项目的`app`目录运行：

```sh
../.venv/bin/python -m my_strategy.web_dashboard.scripts.serve_dashboard --port 8124
```

方案及20项冻结均线特征定义：[ADR_20261009_MA_Trend_Model_Profile.md](ADR_20261009_MA_Trend_Model_Profile.md)。

## 修改范围

- `services/czsc_research_profiles.py`、`czsc_research_features.py`：新增20项均线趋势输入和独立120日质量/60活跃观察资格。
- `services/czsc_research.py`、`czsc_research_ml.py`：均线数据集、MLP、固定逻辑回归对照、真实训练/推理计数。
- `services/czsc_research_runtime.py`、`czsc_research_models.py`、`storage/czsc_model_releases.py`：逐日路由、独立规则/ML资格、连续账本、缓存及模型身份、正式发布时间边界。
- `services/czsc_analysis.py`、`configs/czsc_research.json`、网页`index.html`及`workbench.js`：接通默认均线模型并显示真实设备工作及影子概率。
- `scripts/verify_czsc_research_reproducibility.py`及相关测试：按冻结实际设备验证，覆盖MPS/CPU、旧模型兼容及缓存身份。
- `app/AGENTS.md`、方案和本运行记录：记录授权、模型契约、实施及验证证据。原生CZSC和vendor未改动。

源码证据：[source-changes.json](../../knowledge_base_system/validation/20261009-110834-6355e11c/source-changes.json)、同目录 `manual-source.patch.gz`。工作区无Git；本地96文件基线记录的是root集成前状态，部分并行初始特征改动早于该快照，单独保存最终哈希，不能声称完整Git差异。

## 边界

不调用promote，不建立正式发布事件。即使重建窗口通过门槛，也不能替代模型实际提前训练并冻结后的独立正式认证。fresh/risk实际退出契约与固定10日目标不同，保留影子。

当前模型/成交支持沪深主板，其他板块仍计入准备覆盖并记录排除。原始行情未复权；历史ST、公司行动和退市股票覆盖存在边界，现有股票池有存活偏差，行业元数据为空。15只较早结束行情不得补替。账本为各股票等额独立账户汇总，不能解释为共享现金实盘组合。
