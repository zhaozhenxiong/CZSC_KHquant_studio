"""Install the CZSC workbench and its frozen attachment runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import venv

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app"


def read_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return {key.strip(): value.strip().strip('"').strip("'") for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#") and "=" in line
            for key, value in [line.split("=", 1)]}


def resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def portable(path: Path, base: Path) -> str:
    return "./" + path.relative_to(base).as_posix() if path.is_relative_to(base) else path.as_posix()


def nvidia_available() -> bool:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    try:
        result = subprocess.run([executable, "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5, check=False)
        return result.returncode == 0 and bool(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


def torch_profile(device: str) -> str:
    """Choose an installation backend independently of runtime device selection."""
    system, machine = platform.system(), platform.machine().lower()
    if device.startswith("cuda"):
        if system not in {"Windows", "Linux"}:
            raise ValueError("CUDA 安装仅支持 Windows/Linux；Apple Silicon 请使用 --device mps 或 auto")
        return "cu130"
    if device == "mps" and (system != "Darwin" or machine not in {"arm64", "aarch64"}):
        raise ValueError("MPS 安装需要 Apple Silicon macOS 与原生 ARM64 Python 3.12")
    if system == "Darwin":
        if machine not in {"arm64", "aarch64"}:
            raise ValueError("当前 macOS 发布环境支持 Apple Silicon ARM64；请使用原生 ARM64 Python 3.12")
        return "mps"
    if device == "auto" and nvidia_available():
        return "cu130"
    return "cpu"


def source_tree_hash(manifest: dict) -> str:
    return hashlib.sha256("\n".join(f"{item['path']} {item['sha256']}" for item in manifest["files"]).encode()).hexdigest()


def read_wheel_index(path: Path, manifest: dict) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("schema_version") != 1 or payload.get("package") != "czsc"
            or payload.get("version") != manifest["version"]
            or payload.get("source_tree_sha256") != source_tree_hash(manifest)):
        raise ValueError("CZSC wheel 索引与冻结源码不匹配")
    return payload


def compatible_wheel(executable: Path, directory: Path, manifest: dict) -> Path | None:
    """Trust only frozen provenance, then match the target interpreter's wheel tags."""
    index = directory / "wheel-index.json"
    if index.is_file():
        payload = read_wheel_index(index, manifest)
        entries = payload["wheels"]
    else:
        entries = [{"filename": manifest.get("built_wheel_name", "missing.whl"),
                    "sha256": manifest.get("built_wheel_sha256")}]
    for entry in entries:
        name = entry["filename"]
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("CZSC wheel 索引含非法路径")
    entries = [entry for entry in entries if (directory / entry["filename"]).is_file()]
    if not entries:
        return None
    # The freshly created venv supplies pip; the bootstrap Python need not have pip.
    script = ("import json,sys; from pip._vendor.packaging.tags import sys_tags; "
              "from pip._vendor.packaging.utils import parse_wheel_filename; "
              "wheels={name:parse_wheel_filename(name) for name in json.loads(sys.argv[1])}; "
              "print(json.dumps({'target_tags':[str(tag) for tag in sys_tags()], "
              "'wheels':{name:{'package':str(value[0]),'version':str(value[1]),'tags':[str(tag) for tag in value[3]]} "
              "for name,value in wheels.items()}}))")
    metadata = json.loads(subprocess.check_output([str(executable), "-c", script,
                                                   json.dumps([entry["filename"] for entry in entries])], text=True))
    target_tags = metadata["target_tags"]
    candidates = []
    for entry in entries:
        name = entry["filename"]
        path = directory / name
        if not path.is_file():
            continue
        parsed = metadata["wheels"][name]
        if parsed["package"] != "czsc" or parsed["version"] != manifest["version"]:
            raise ValueError("CZSC wheel 名称与冻结版本不匹配")
        matching = set(parsed["tags"]).intersection(target_tags)
        if not matching:
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"冻结 CZSC wheel 校验失败：{name}")
        if index.is_file() and (entry.get("build", {}).get("cargo_locked") is not True
                               or entry.get("build", {}).get("source_algorithm_patches") is not False):
            raise ValueError(f"CZSC wheel 缺少冻结构建来源：{name}")
        candidates.append((min(target_tags.index(tag) for tag in matching), name, path))
    return min(candidates)[2] if candidates else None


def record_wheel_provenance(wheel: Path, manifest: dict) -> None:
    """Retain verified external wheel evidence where runtime doctor reads it."""
    source = wheel.parent / "wheel-index.json"
    destination = APP / "vendor/wheels/wheel-index.json"
    if not source.is_file() or source.resolve() == destination.resolve():
        return
    external = read_wheel_index(source, manifest)
    selected = next((entry for entry in external["wheels"] if entry["filename"] == wheel.name), None)
    if (selected is None or hashlib.sha256(wheel.read_bytes()).hexdigest() != selected["sha256"]
            or selected.get("build", {}).get("cargo_locked") is not True
            or selected.get("build", {}).get("source_algorithm_patches") is not False):
        raise ValueError("外部 CZSC wheel 构建来源校验失败")
    local = read_wheel_index(destination, manifest) if destination.is_file() else {**external, "wheels": []}
    local["wheels"] = [entry for entry in local["wheels"] if entry["filename"] != wheel.name] + [selected]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(local, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def install(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="安装完整 KHQuant CZSC 工作台（Python 3.12，CPU/CUDA/MPS）")
    parser.add_argument("--venv-dir")
    parser.add_argument("--data-dir")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--raw-db")
    parser.add_argument("--port", type=int)
    parser.add_argument("--upgrade", action="store_true", help="重新安装当前CZSC环境的冻结依赖")
    parser.add_argument("--gpu", action="store_true", help="兼容旧参数；完整安装默认包含 PyTorch 并自动选择平台")
    parser.add_argument("--device", help="计算设备：auto、cpu、cuda、cuda:编号 或 mps")
    parser.add_argument("--wheel-dir", help="已下载的平台 CZSC wheel 与 wheel-index.json 目录")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 12):
        parser.error("当前发布环境需要 Python 3.12；其他版本尚未验收")
    previous = read_env(ROOT / ".env")
    device = args.device or ("auto" if args.gpu else previous.get("KHQUANT_COMPUTE_DEVICE", "auto"))
    if not re.fullmatch(r"auto|cpu|mps|cuda(?::\d+)?", device):
        parser.error("计算设备必须为 auto、cpu、cuda、cuda:编号 或 mps")
    try:
        profile = torch_profile(device)
    except ValueError as exc:
        parser.error(str(exc))
    project = resolve(previous.get("KHQUANT_PROJECT_ROOT", "./app"), ROOT)
    env_dir = resolve(args.venv_dir or previous.get("KHQUANT_VENV_DIR", "./.venv"), ROOT)
    data = resolve(args.data_dir or previous.get("KHQUANT_DATA_ROOT", "./my_strategy/data"), project)
    artifacts = resolve(args.artifact_dir or previous.get("KHQUANT_ARTIFACT_ROOT", "./my_strategy/artifacts"), project)
    raw = resolve(args.raw_db or previous.get("KHQUANT_RAW_DB", data / "raw" / "khquant_raw.db"), project)
    port = args.port if args.port is not None else int(previous.get("KHQUANT_DASHBOARD_PORT", 8124))
    if not 1 <= port <= 65535:
        parser.error("端口必须为1至65535")
    executable = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    wheels = resolve(args.wheel_dir or APP / "vendor" / "wheels", ROOT)
    torch_requirements = f"app/requirements-czsc-torch-{profile}.lock"
    config = {"venv": str(env_dir), "data": str(data), "raw_db": str(raw), "artifacts": str(artifacts), "port": port,
              "requirements": "app/requirements-czsc-runtime.lock", "source": "app/vendor/czsc",
              "torch_requirements": torch_requirements, "torch_profile": profile,
              "gpu_requirements": torch_requirements if profile == "cu130" else None,
              "compute_device": device, "wheel_dir": str(wheels), "python": "3.12"}
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    if not executable.is_file():
        venv.EnvBuilder(with_pip=True).create(env_dir)
    env = dict(os.environ, PYTHONPATH=str(APP), PYTHONDONTWRITEBYTECODE="1")
    if sysconfig.get_platform() == "win-amd64":
        env.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")
    env.pop("PYTHONHOME", None)

    def run(*command: str) -> None:
        subprocess.run([str(executable), *command], cwd=APP, env=env, check=True)

    manifest_path = APP / "czsc-source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        file = APP / "vendor" / "czsc" / entry["path"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"冻结CZSC源码校验失败：{entry['path']}")
    run("-m", "pip", "--isolated", "install", "-r", str(APP / "requirements-czsc-runtime.lock"))
    run("-m", "pip", "--isolated", "install", "-r", str(ROOT / torch_requirements),
        "--constraint", str(APP / "requirements-czsc-runtime.lock"))
    wheel = compatible_wheel(executable, wheels, manifest)
    if wheel:
        run("-m", "pip", "--isolated", "install", "--no-deps", "--force-reinstall", str(wheel))
        record_wheel_provenance(wheel, manifest)
    else:
        if not shutil.which("cargo"):
            raise RuntimeError("缺少匹配当前平台的 CZSC wheel。请下载 Release 的平台 wheel 与 wheel-index.json，使用 --wheel-dir 指定；或安装 Rust 与本机编译器后从冻结源码构建。")
        print("未找到匹配平台的 CZSC wheel，使用本机 Rust/编译器构建冻结源码。")
        run(str(APP / "build_czsc_runtime.py"), "--out", str(ROOT / ".runtime-wheels"))
    run("-m", "pip", "--isolated", "install", "--no-deps", "--no-build-isolation", "--editable", str(ROOT))
    run("-m", "pip", "check")
    for directory in (data / "raw", data / "processed" / "czsc", data / "metadata", data / "logs", artifacts / "runs"):
        directory.mkdir(parents=True, exist_ok=True)
    # Retain provider credentials and unrelated settings; retire obsolete KHQuant paths.
    obsolete = {"KHQUANT_PROCESSED_DB", "KHQUANT_OUTPUT_ROOT", "KHQUANT_STORAGE_MODE", "KHQUANT_SERVICE_LEGACY_ROUTES", "KHQUANT_SERVICE_LEGACY_SUNSET", "KHQUANT_SERVICE_JOB_HEARTBEAT_S"}
    values = {key: value for key, value in previous.items() if key not in obsolete}
    values.update(KHQUANT_ENV_SCHEMA="2", KHQUANT_PROJECT_ROOT="./app", KHQUANT_VENV_DIR=portable(env_dir, ROOT),
                  KHQUANT_DATA_ROOT=portable(data, APP), KHQUANT_RAW_DB=portable(raw, APP),
                  KHQUANT_ARTIFACT_ROOT=portable(artifacts, APP), KHQUANT_LOG_ROOT=portable(data / "logs", APP),
                  KHQUANT_METADATA_ROOT=portable(data / "metadata", APP), KHQUANT_DASHBOARD_PORT=str(port), PYTHONPATH="./app", TZ="Asia/Shanghai")
    values.update(KHQUANT_COMPUTE_DEVICE=device, KHQUANT_CPU_WORKERS=previous.get("KHQUANT_CPU_WORKERS", "4"),
                  KHQUANT_GPU_BATCH_SIZE=previous.get("KHQUANT_GPU_BATCH_SIZE", "32"))
    temporary = ROOT / ".env.tmp"
    temporary.write_text("\n".join(f'{key}="{value}"' for key, value in values.items()) + "\n", encoding="utf-8")
    temporary.replace(ROOT / ".env")
    env.update(values)
    run("-m", "my_strategy.cli", "doctor", "--check-write", "--strict", "--json")
    print(f"CZSC安装完成。启动：{executable} -m my_strategy.cli dashboard")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(install())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"安装失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
