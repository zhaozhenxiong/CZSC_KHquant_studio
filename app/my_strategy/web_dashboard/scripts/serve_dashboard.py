"""Uvicorn entrypoint."""

from __future__ import annotations

import os
import argparse

import uvicorn

from my_strategy.web_dashboard.config import DEFAULT_HOST, DEFAULT_PORT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KHQuant Web Dashboard server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get("KHQUANT_API_TOKEN"):
        raise ValueError("非本机监听须配置 KHQUANT_API_TOKEN")
    uvicorn.run(
        "my_strategy.web_dashboard.app:create_app",
        host=args.host,
        port=args.port,
        factory=True,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
