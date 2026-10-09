"""CZSC workbench application; no legacy API or result fallback."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
import os
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from my_strategy.web_dashboard.api import router
from my_strategy.web_dashboard.config import STATIC_ROOT
from my_strategy.web_dashboard.tasks import TaskManager
from my_strategy.storage.personal_portfolio import PersonalStore


def create_app(task_manager: TaskManager | None = None, personal_store: PersonalStore | None = None) -> FastAPI:
    manager = task_manager or TaskManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await asyncio.to_thread(manager.close)

    app = FastAPI(title="KHQuant CZSC 结构研究", version="1.0.1", lifespan=lifespan)
    app.state.tasks = manager
    app.state.personal = personal_store or PersonalStore(manager.db_path.parent / "personal_portfolio.db")

    @app.middleware("http")
    async def security(request: Request, call_next):
        if request.url.path.startswith("/api/") and request.url.path != "/api/health":
            token = os.environ.get("KHQUANT_API_TOKEN", "")
            provided = request.headers.get("authorization", "")
            if token and not hmac.compare_digest(provided, f"Bearer {token}"):
                return JSONResponse({"detail": "连接密钥无效"}, status_code=401)
            origin = request.headers.get("origin")
            if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
                if urlsplit(origin).netloc != request.headers.get("host"):
                    return JSONResponse({"detail": "拒绝跨来源写请求"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'self'"
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "找不到指定任务或运行"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def unavailable(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

    @app.get("/")
    @app.get("/analysis")
    @app.get("/scan")
    @app.get("/backtest")
    @app.get("/data")
    @app.get("/watchlist")
    @app.get("/holdings")
    def shell():
        return FileResponse(STATIC_ROOT / "index.html", headers={"Cache-Control": "no-store"})

    return app
