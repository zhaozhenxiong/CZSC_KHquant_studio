"""Add frozen CZSC provenance resources to the application wheel."""
import hashlib
import json
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app"


class BuildPy(build_py):
    def run(self):
        super().run()
        manifest_path = APP / "czsc-source-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        destination = Path(self.build_lib) / "my_strategy" / "_runtime"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, destination / manifest_path.name)
        for entry in manifest["files"]:
            relative = Path("vendor/czsc") / entry["path"]
            source = APP / relative
            if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Frozen CZSC source mismatch: {entry['path']}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        index = APP / "vendor/wheels/wheel-index.json"
        if index.is_file():
            target = destination / "vendor/wheels/wheel-index.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(index, target)


requirements = [line for line in (APP / "requirements-czsc-runtime.lock").read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
setup(cmdclass={"build_py": BuildPy}, install_requires=[*requirements, "czsc==1.0.1", "torch==2.13.0"])
