"""CZSC workbench static and server settings."""
import os
from pathlib import Path

STATIC_ROOT = Path(__file__).resolve().parent / "static"
DEFAULT_HOST = os.environ.get("KHQUANT_DASHBOARD_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("KHQUANT_DASHBOARD_PORT", "8124"))
