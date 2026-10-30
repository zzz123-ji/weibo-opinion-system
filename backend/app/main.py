"""FastAPI 入口。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import REPO_ROOT, settings
from .db import SessionLocal, init_db
from .routers import auth, config, crawl, eval, hotsearch, labels, posts, sentiment, trends
from .schemas import HealthOut
from .seed import ensure_admin, seed_demo

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        ensure_admin(db)
        if settings.seed_demo:
            seed_demo(db, settings.demo_csv)
    finally:
        db.close()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="微博舆情采集、趋势、标注与情感引擎评估系统",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in (
        auth.router,
        posts.router,
        trends.router,
        hotsearch.router,
        labels.router,
        eval.router,
        config.router,
        crawl.router,
        sentiment.router,
    ):
        app.include_router(router, prefix=settings.api_prefix)

    @app.get(f"{settings.api_prefix}/health", response_model=HealthOut, tags=["system"])
    def health():
        from .services.worker import active_engine

        db = SessionLocal()
        try:
            engine_name = active_engine(db)
        finally:
            db.close()
        return HealthOut(
            status="ok",
            app=settings.app_name,
            database="postgresql" if not settings.is_sqlite else "sqlite",
            active_engine=engine_name,
            version=__version__,
        )

    # 生产构建后的 Vue 静态文件；开发时由 Vite 提供。
    dist = REPO_ROOT / "frontend" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")

    return app


app = create_app()
