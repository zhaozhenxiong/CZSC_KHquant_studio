# Apple Silicon Mac

使用原生 ARM64 Python 3.12，在仓库根执行：

~~~bash
python3.12 install.py --device auto
./khquant-native.sh dashboard
~~~

可以明确选择 --device mps 或 --device cpu。MPS 加速当前 PyTorch 训练与推理，结构和账本保持精确 CPU 路径。缺匹配原生 CZSC wheel 时需要 Rust 和 Xcode command-line tools；平台 Release 会携带经来源校验的预构建 wheel。

真机验收：

~~~bash
.venv/bin/python app/verify_installation.py --device mps --output installation-mps.json
~~~

MPS 检测与真实算子执行都须成功；不以 Windows 测试或普通 macOS CPU CI 代替 MPS 真机结果。安装、数据包和升级说明见仓库根 INSTALL_NATIVE.md。
