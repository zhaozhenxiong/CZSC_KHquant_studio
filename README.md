# CZSC_KHquant_studio
# KHQuant · CZSC 与量价研究工作台

六个网页入口：结构分析、市场扫描、策略回测、我的关注、我的持仓、数据与任务。当前组合研究使用冻结 CZSC 1.0.1、均线量价和 PyTorch；支持行情更新、研究训练、模型推理、按日期路由及状态打包。

本项目的主要缠论结构分析基于 [CZSC](https://github.com/waditu/czsc)，使用随项目冻结的 CZSC 1.0.1 源码，包括分型、笔、中枢及日线、周线、月线多周期结构。感谢 CZSC 项目作者及所有贡献者提供的开源实现。

## 安装和启动

安装原生 Python **3.12**，克隆仓库或解压 Release。首次安装默认创建 .venv；升级保留已有 .env 配置和环境路径。

Windows x64：

~~~powershell
py -3.12 install.py --device auto
.\khquant-native.bat dashboard
~~~

Apple Silicon Mac（原生 ARM64 Python）：

~~~bash
python3.12 install.py --device auto
./khquant-native.sh dashboard
~~~

Linux x64：

~~~bash
python3.12 install.py --device cpu
./khquant-native.sh dashboard
~~~

浏览器打开 <http://127.0.0.1:8124>。安装包含 PyTorch 研究依赖；auto 在 NVIDIA 环境选择 CUDA，在 Apple Silicon 选择 MPS，其余选择 CPU。可以明确使用 --device cpu、--device cuda:0 或 --device mps。明确请求不可用的设备会报错。

安装器只接受经冻结源码校验的 CZSC wheel。Release 应携带目标平台 wheel 和 wheel-index.json；可通过 --wheel-dir 使用单独下载的资产。当前本地产物只携带 Windows wheel，Mac/Linux 首个原生 wheel 必须由相应平台 CI 构建；源码安装缺 wheel 时需要 Rust 与本机编译工具。平台测试状态见 runtime-manifest.json，MPS 正式验收必须使用真实 Mac。

## 首次使用

代码包不包含私人行情库、训练模型、关注或持仓。首次运行可用“数据与任务”下载行情，或安装时指定 --raw-db。研究训练需要已核验的交易日历；可导入包含日历和模型依赖的状态包。软件安装完成与行情、模型资源就绪分别显示。

激活环境后，khquant 与 python -m my_strategy.cli 共用入口：

~~~bash
khquant doctor --check-write --strict --json
khquant update-data --mode local --stocks 600519.SH
khquant analyze --symbol 600519.SH --json
khquant scan --stocks 600519.SH,000001.SZ --device auto --json
khquant backtest --symbol 600519.SH --start 2026-01-01 --device auto --json
khquant research-train --end YYYY-MM-DD --calendar-run-id VERIFIED_CALENDAR_RUN --device auto
~~~

网页、CLI 和模型推理按本机选择运行设备，模型原训练设备只保留为历史来源记录。MPS 加速神经网络训练和推理；精确结构、价格比较及成交账本由 CPU 执行。任务结果区分设备可用性和实际工作。

## 分发和迁移

源码 Release，不含业务数据和个人记录：

~~~bash
python app/build_release.py --output-dir dist/releases/0.1.0
~~~

在源码仓库或解压的源码 Release 环境中打包研究状态，目标目录须不存在：

~~~bash
khquant package --target ../khquant-state --include-data --include-artifacts all
~~~

正式 wheel 安装只提供运行代码，不能从 site-packages 导出源码包。先克隆仓库或解压源码 Release，再用 `--from-project /path/to/KHQuant` 指定来源。导出状态采用该来源目录的路径配置；迁移 wheel 的现有状态时，先将来源目录的 `.env` 指向实际数据、模型 artifacts 和 metadata 目录。

需要自己迁移关注与手动持仓时，显式追加 --include-personal。模型、核验日历及发布证据按依赖完整携带；模型哈希、历史时间和影子资格保持不变。安装方式及验收详见 [INSTALL_NATIVE.md](INSTALL_NATIVE.md)。

## 研究边界

默认组合研究；--structure-only 选择原生结构对照。收盘信号在下一可执行交易日开盘尝试成交，费用、拒单、每日净值和期末持仓均留存。多股回测采用固定等额独立账户。当前使用未复权行情，历史 ST、分红送转等限制随结果披露。

训练候选只有通过独立、完整股票池的费用后账本门槛才可晋升。目录名 production 不代表已正式发布；迁移和安装也不会赋予发布资格。旧 V2/Z1 模型与派生仓库已退役。
