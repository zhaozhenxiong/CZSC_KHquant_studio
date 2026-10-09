# KHQuant 0.1.1 验证

2026-10-10，Apple Silicon macOS ARM64 / Python 3.12。

- 全量源码语法检查通过；817 项测试通过，7 项按平台条件跳过。迁移测试使用合法特征 profile，并统一 macOS 临时目录真实路径。
- 独立安装主程序 wheel，确认应用来自新环境 site-packages，排除源码导入；复用本机既有锁定依赖和冻结原生 CZSC，未执行全新依赖下载验收。
- CPU 与真实 MPS 冒烟均通过：严格 doctor、训练保存与重新加载、各 40 行实际推理、六个网页入口、个人关注及持仓跨重启保存。
- 226 项冻结 CZSC 源码逐项 SHA 校验通过；MA/dual/W 服务、配置与网页资源完整；无行情、模型权重或私人状态。
- Windows、Linux 和 CUDA 本版验收尚未执行；完整新嵌套模型状态迁移未宣称可用。模型仍为影子，以上工程验证不认证市场收益。

主程序 wheel SHA-256：`ffa2239c327e5c73b5fc8a986eac8429a08dc95424c20a1bfc2a9bd8dda04834`。

最终源码 ZIP 的版本、合并提交及完整性信息由 build_release.py 从 main 重新生成，见 release.json、image-manifest.json 与 SHA256SUMS.txt。原始安装报告与回归日志保留于原本地工作区，本文记录可公开的范围与结果。
