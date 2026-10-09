---
type: operation
date: 2026-10-08
updated: 2026-10-09
---

# 免费未复权历史行情下载契约

首次空库通过 `RawMarketRepository.connect()` 建立 raw schema，然后冻结股票池并运行普通 `update-data`。默认历史起点为 2015-01-01；空库使用 local 模式时必须明确股票列表。运行不使用 `--today-realtime`，也不将实时源作为历史回退。

## Sina：sina_unadjusted_v1

- 固定请求 `https://finance.sina.com.cn/realstock/company/{sh|sz|bj}{code}/hisdata_klc2/klc_kl.js`，curl 请求上限 20 秒，外层进程上限 25 秒。
- 使用安装的 AkShare `hk_js_decode` 解码原始 `KLC_K2_{symbol}` 历史数据；只执行安装的解码器，远端内容仅作为字符串参数。不请求前/后复权因子。
- 原始 OHLC 同日复制到 `trade_open/high/low/close`；volume 单位为股，amount 单位为元。turnover 未取股本数据，暂为 0，不能理解为真实零换手。
- 保留原始行情日期并裁剪请求区间，不做股本日期外连接、不前向填充、不按相同 OHLCV 删除合法日期。原始缺失日仍缺失。重复日期、非有限值、非法 OHLC、负量额均拒绝。
- MiniRacer 在本机并发首次初始化时曾触发原生崩溃；仅解码阶段加锁，HTTP 下载仍可并发。

真实样本：600519.SH 与 000001.SZ 均得到 2856 条，2015-01-05 至 2026-10-08，`trade_*` 完整。920000.BJ 原始解码有两条非法历史行情：2019-05-23 和 2019-07-12 的 OHLC 为零，成交量/金额为正；引入下述 BSE 上市边界前，严格入口因此返回失败。两条 NEEQ 前史异常仍作为原始证据保留，不补价、不修改原始响应。AkShare 普通 `stock_zh_a_daily(adjust='')` 曾返回 1401 条，但它会使用股本日期 outer join、ffill 与价格去重，不能视作本契约的原始完整证据。

茅台原始压缩响应证据：93590 字节，SHA256 `1c0b0f70fefb00fea48c773b395688650456cd6e12a4f5129fc9aa643be22729`。最新原始 bar：2026-10-08，close 1255.79，volume 2516757 股，amount 3145413157 元。

2026-10-09 对本次下载失败的三只科创板股票复核：新浪均在 2024-11-06 返回一根 open/high/low 为 0、close 为正、volume/amount 为 0 的非法占位 bar。三股的 close 分别为 `688089.SH: 20.92`、`688143.SH: 27.58`、`688173.SH: 11.80`，各自只有这一根 OHLC 异常。它们处于请求的有效历史区间内，严格入口均返回 `invalid historical OHLC range`；不得删除坏 bar 后把新浪结果宣称为通过，也不得把 close 复制到 open/high/low 修价。三份原始 JS、解码结果和失败报告均保存在下述腾讯科创板核验快照中。

## Eastmoney：eastmoney_unadjusted_v1

固定日线接口 `https://push2his.eastmoney.com/api/qt/stock/kline/get`，`klt=101`、`fqt=0`，`secid=1.code` 用于上海、`0.code` 用于深圳/北京。请求的字段列表保留字面逗号。

| 字段 | 日期 | 开 | 收 | 高 | 低 | 量 | 额 | 振幅 | 涨跌幅 | 涨跌额 | 换手率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fields2 | f51 | f52 | f53 | f54 | f55 | f56 | f57 | f58 | f59 | f60 | f61 |

响应必须匹配请求股票代码，且每根 kline 恰有 11 字段。成交量手乘 100 转为股，金额维持元；同日未复权 OHLC 复制 `trade_*`。同样拒绝格式、日期、重复和价格错误。

本次茅台 curl 曾一次成功返回 2856 条全历史：2015-01-05 至 2026-10-08，242477 字节，SHA256 `905679d0afdfaeecb9372bfbe0fdd5df5801c2aac38e345b8ba9d55181aec2fa`。与新浪样本的 close 相同，量比约为 100，金额近似相同，确认单位转换。随后沪深京请求重复出现 curl 52（Empty reply），所以此响应仅验证数据契约，不能证明当前网络持续可用；运行时错误必须保留，不可用复权或实时报价顶替。

## Tencent 科创板：tencent_star_unadjusted_v1

该源仅支持显式 `.SH` 且六位代码以 `688` 开头的科创板股票；其他代码返回 unsupported，不扩大到沪市主板、深圳或北京。来源精确名称为 `tencent_star_unadjusted_v1`，进入研究数据允许源清单；历史更新回退顺序位于新浪之后、付费 `fuyao`/`tushare` 之前。新配置由下一个重试进程读取，已经启动的下载任务不会自动换源，本次修改与核验不更改其进程或业务数据库。

- 固定接口 `https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get`，逐年请求 `param=sh{code},day,{year}-01-01,{year}-12-31,640,`。末项必须保留为空，表示未复权；只读取匹配证券的纯 `day` 字段，不接受 `qfqday`/`hfqday`，不从同一响应的 `qt` 实时报价读取或补替历史。
- 原始 `day` 行依次提供日期、open、close、high、low、volume；volume 已为股，直接保留。amount 取行中第 9 项（从 1 开始计数），原单位万元，乘 10000 转为元；原接口保留两位小数，金额精度为 100 元。原始未复权 OHLC 同日复制到 `trade_open/high/low/close`，不执行复权、价格填充或日期外连接。
- 实际 API 会忽略起始日期，返回目标年前的历史，常见为截至指定年末的最近 640 根。后端必须对每页按该目标年及最终请求范围裁剪，再按日期合并；每个涉及的年份都要独立请求 640 根。达到上限且历史起点仍晚于目标窗口起点的页属于可能截断，必须失败；单次最后 320/640 根不能冒充完整历史。
- HTTP/JSON/API 错误、缺失或错误证券字段、复权字段、非法日期、非法 OHLC、非有限值、负量额与截断均明确拒绝。每页重复日期必须拒绝；逐年响应重叠日期的历史修订冲突也必须拒绝，不能由后页静默覆盖前页。请求前上市年份返回空 `day` 本身不代表请求失败；所有年份合并仍没有请求区间内行情时返回失败。

真实核验使用 2015 至 2026 共 12 个年份，三股共 36 次请求；现场探测的 2026 尾年结束参数按本次最终截止收紧为 `2026-10-08`，其余年份均为 `12-31`。退出码和 API code 均为 0，所有价格字段均为 `day`，没有请求失败、目标年截断、重复日期或非法 bar。原始响应可能有年际重叠，但按目标年裁剪后的完整区间如下，实际数据截止 2026-10-08：

| 股票 | 腾讯合法历史行数 | 实际范围 | 新浪唯一额外日期 |
| --- | ---: | --- | --- |
| 688089.SH | 1635 | 2019-12-19 至 2026-10-08 | 2024-11-06 非法占位 |
| 688143.SH | 912 | 2022-12-12 至 2026-10-08 | 2024-11-06 非法占位 |
| 688173.SH | 1129 | 2022-01-21 至 2026-10-08 | 2024-11-06 非法占位 |

除新浪上述一根非法占位外，三股两源日期集合完全一致；全部共同日期的 OHLC 和 volume（股）逐根完全相同。腾讯金额经万元转元后与新浪金额的最大绝对差约为 50 元，符合其原字段 100 元精度，但不能宣称 amount 与新浪逐元相同。该比较核验的是这三个样本，不能自动证明全部科创板股票、所有未来请求或网络持续可用。

腾讯在 2024-11-06 均没有交易 bar。当前证据没有官方停牌核验，不能据此断言三股实际停牌。回退取得独立来源真实返回的历史，保留该日缺失；这既不是补回新浪异常日的价格，也不豁免下游交易日历与缺失覆盖核验。新浪错误仍作为失败来源证据保留。

持久证据目录为 `my_strategy/data/metadata/tencent_star_history_verification_20261009/`，所有文件由现场探测目录逐字节复制，没有重新抓取或修订响应：

- [sha256_manifest.json](../../data/metadata/tencent_star_history_verification_20261009/sha256_manifest.json)：61 个文件，共 2,866,650 字节，列出相对路径、字节数及逐文件 SHA256；清单自身 SHA256 为 `9d8d8c2ae8f717145a10da9f4bc9bf3b0a1327f89fb692dd902b0f36780a9762`。
- [tencent-yearly/requests.json](../../data/metadata/tencent_star_history_verification_20261009/tencent-yearly/requests.json)：36 个精确 URL、退出码、API code、返回字段、原始日期范围、原始响应 SHA256 与裁剪到目标年的 `day` 行；同目录 `{code}-{year}.txt` 保存每页原件，`{code}-full-day.json` 保存按目标年裁剪合并结果。requests 清单 SHA256 为 `9d402178cbffc8eb17a9cafcdd12b8862dd6e8a4b3c451c1d73190e4746c8b88`，36 页原始 txt 的哈希均与清单一致。
- [tencent-yearly/summaries.json](../../data/metadata/tencent_star_history_verification_20261009/tencent-yearly/summaries.json)：行数、覆盖、日期/价格/量差异、金额精度及目标日缺失报告；SHA256 为 `bd108b4997678535c5d07bc1be36e8d7b0ba958e24fea9399408048759ae5f9e`。
- `sh{code}.js`、`sh{code}-decoded.json`、`sh{code}-report.json`：新浪原始响应、解码结果及严格失败报告。三股原始 JS SHA256 依次为 `eeb6fc251f8e7c953f33b260b0ff50b8448c518b45f36d465a8329c5ecc8b4d0`、`60242399340144affbe8c1a5a77df576cd208b8a483a31106bb0600d269a59ed`、`cfb929039d2100a5e1f3dacdd2ffb10fdf809c9aed24c195a604fcd971eca658`。
- [alternate-source-manifest.json](../../data/metadata/tencent_star_history_verification_20261009/alternate-source-manifest.json) 与 [tencent-2024-manifest.json](../../data/metadata/tencent_star_history_verification_20261009/tencent-2024-manifest.json)：保留初次腾讯单次/2024 区间探测与东方财富空响应记录，以及各自原始 `.txt`。单次大范围探测的错误响应不是全历史证据；完整性依据是上述逐年请求与合并核验。

## 北交所上市边界与异常前史

2026-10-08 的 `akshare.stock_info_bj_name_code`（AkShare 1.19.1）经北交所 `nqxxController/nqxxCnzq.do` 接口取得 348 只股票，代码唯一，`fxssrq` 与 `xxgprq` 全部相同且日期完整。接口日期包含原全国股转系统精选层挂牌日期，不能直接把它当作每股 BSE 上市日。北交所 [本所简介](https://www.bse.cn/company/introduce.html) 明确开市交易日期为 2021-11-15；因此本项目的 BSE 有效起点按 `max(api fxssrq, 2021-11-15)` 推导。348 只中有 67 只的接口日期早于开市日，边界抬至 2021-11-15；其余 281 只沿用接口日期。该规则属于派生边界，逐股保留原值与推导标志，并非宣称接口直接提供了每股 BSE 上市日期。

安徽凤凰 `920000.BJ`（原代码 `832000`）的接口日期为 2020-12-23，BSE 有效起点为 2021-11-15；[公司 2023 年年度报告](https://www.bse.cn/disclosure/2024/2024-04-26/56bde047e6964582a56ab5dd56f3129a.pdf) 将上市时间列为 2021-11-15。2019-05-23 和 2019-07-12 两条零 OHLC、有正成交量/金额的 NEEQ 原始异常在 BSE 有效区间之外，原始证据继续保留，不能补填价格。历史入口按请求区间及已核验 BSE 边界选取有效范围，然后严格核验其中每根 bar；真正 BSE 有效区间内的非法 OHLC、重复日期、非有限值、负量额仍必须明确失败，禁止借边界裁剪静默删除区间内坏 bar。

接入快照后的新浪真实验收（2026-10-09 00:02，本地数据截止 2026-10-08）：请求 `920000.BJ` 的 2015-01-01 至 2026-10-08 历史，原始解码 1410 根；排除 BSE 边界前 224 根后返回 1186 根，实际范围 2021-11-15 至 2026-10-08。返回日期与原始响应的全部上市后日期集合完全一致，逐根 `trade_*` 与同日 OHLC 相同，BSE 区间内非法 OHLC 数为 0。原始 JS SHA256 `beeb8a831499056b0d2214a34f2e4fe0cad2ade359f6a705e131d56a3444e2e4`；验收导出 CSV SHA256 `ceaca5e6b7cb85e6fa22a7c9799cc3a8cb074a8a0dbd0f54758eb6f144a89e57`。原始 2019-05-23 异常的成交量为 449000 股、金额 767790 元，2019-07-12 为 150000 股、450000 元，两条 OHLC 均为零；这些数值继续作为边界前原始证据保留，验收没有修价、填补或写入业务数据库。

持久快照位于 `my_strategy/data/metadata/`：

- [bse_listing_boundaries_20261008.json](../../data/metadata/bse_listing_boundaries_20261008.json)：`schema_version=1`，含 348 条逐股记录，保留原日期、派生规则及官方来源。`source_snapshot.file` 和逐股 `source.raw_api_snapshot` 使用同目录相对文件名；`source_snapshot.sha256` 绑定原始 API 文件。SHA256 为 `05fb0ff274ed85cd2d06d86e0fbfb50a32e7f58143a998e318a1dbfba7564c08`。
- [bse_stock_info_raw_api_20261008.json](../../data/metadata/bse_stock_info_raw_api_20261008.json)：原文件逐字节复制，SHA256 为 `b1dea12c1f7c66c552e34dea32269173841ea264dedc2462fd0699ce4dfc5a1a`。文件保存 19 次分页响应、共 368 条响应记录；初次探测的第 0 页 20 条又在正式分页中保存一次，两个响应完全相同。唯一分页数为 18，唯一证券数仍为 348；重复响应原样保留，并在边界文件 `summary` 标注，不能误解为股票池重复。
- [bse_listing_boundaries_20261008.sha256](../../data/metadata/bse_listing_boundaries_20261008.sha256)：同时列出边界文件和原始 API 文件哈希，供独立完整性校验。运行时直接用边界快照的 `source_snapshot.sha256` 核验其引用的原始 API 文件，不读取侧车中的边界文件哈希。

默认配置在 `my_strategy/configs/data_config.yaml` 的 `sina_unadjusted_v1.listing_boundaries_file` 指向 `my_strategy/data/metadata/bse_listing_boundaries_20261008.json`，按应用根解析。北京标的必须具备有效边界配置、受哈希绑定的原始快照和对应股票记录；配置缺失、文件损坏、哈希不符或日期/推导关系非法时，在联网前明确失败。沪深标的无需此 BSE 快照。

独立历史探测证据为 `/tmp/khquant-bj-api-probes-20261008/manifest.json`，SHA256 `445b2219599cee69499bae865cc246d707f56629247436a8bd303b1e1fa56330`；清单保留精确 URL、退出码、字节数和返回 bars，各响应原件为同目录 `.txt`。腾讯 `fqkline/get`、`newfqkline/get` 都以 `param` 最后一项为空请求未复权日线，并同时探测新旧代码；接口名称中含 `fq` 不能代替实际复权参数与返回字段核验。

| 探测 | 实际结果 | 证据 SHA256 |
| --- | --- | --- |
| 腾讯新代码 920000，2019 区间，两个接口 | 两个响应 `day=[]`；未提供 2019-05-23、2019-07-12 合法历史 bar | `3fda6cd23e4805914c93da4f2088f0c94519ac799b5aabb30c8c394647156057` |
| 腾讯旧代码 832000，2019 区间，newfq 接口 | 返回 7 条旧前史，最后日期 2019-07-04；两条目标日期均缺失 | `9dbe6d471981c3116761985f9cb0abcfb29866c76d7a9f7c4893713b0bff0941` |
| 腾讯新代码 920000，2021-11 请求，newfq 接口 | 实际返回 227 条，2020-12-23 至 2021-11-30；按 BSE 起点截取 12 条，2021-11-15 至 2021-11-30，OHLC 核验通过 | `2c0b7f9713b0efbbe81dda70a84484c8212781d6ef876c1f24191f4f7a5e5670` |
| Eastmoney 新旧代码，`fqt=0`，2019 区间 | 均为 curl 52、0 字节、Empty reply，无可用历史证据 | 空响应哈希均为 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

上述腾讯有效样本可交叉核验 BSE 区间行情，却不能恢复两条缺失的 2019 历史价格；东方财富空响应也不能用于填补。保留接口错误和缺失，不以邻日、后续报价、前向填充或另一交易范围的数据替代。

## 运行与验收

只让明确未复权、含真实 `trade_*` 的源进入历史更新主/回退列表。研究数据允许源清单包含精确名称 `sina_unadjusted_v1`、`eastmoney_unadjusted_v1`、`tencent_star_unadjusted_v1`，这不替代交易日历、缺失覆盖、复权与历史 ST 等其他研究核验。任务完成后检查来源、股票失败数、实际日期范围、`has_trade_price` 与数据质量；失败股票保留明确记录。

离线验证：`test_unadjusted_history_providers.py` 检查请求参数、代码/日期边界、单位、真实同日交易价、非法响应拒绝、相同价格日期保留及下游只读研究核验。测试运行目录由 conftest 隔离，避免读写本机业务库。

发行源码副本说明：此文引用的数据库、模型与审计证据保留于原本地研究工作区，未包含在公开源码或代码发行包中。原始研究记录未改写。
