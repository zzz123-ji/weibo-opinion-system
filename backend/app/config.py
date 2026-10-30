"""运行配置。

所有配置都从环境变量读取，便于 Docker Compose 与本地 SQLite 两种运行方式共用。
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _default_database_url() -> str:
    data_dir = REPO_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(data_dir / 'opinion.db').as_posix()}"


def _default_cookie_key(secret: str) -> str:
    # 把任意字符串派生成 Fernet 需要的 32 字节 urlsafe base64 key。
    digest = hashlib.sha256(f"cookie::{secret}".encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")


@dataclass(frozen=True)
class Settings:
    app_name: str = "微博舆情管理系统"
    api_prefix: str = "/api"
    database_url: str = field(default_factory=_default_database_url)
    jwt_secret: str = field(
        default_factory=lambda: os.getenv(
            "JWT_SECRET", "dev-secret-change-me-please-use-32-bytes-minimum"
        )
    )
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = field(default_factory=lambda: _int("ACCESS_TOKEN_MINUTES", 720))
    cookie_key: str = field(
        default_factory=lambda: os.getenv("COOKIE_ENCRYPTION_KEY")
        or _default_cookie_key(
            os.getenv("JWT_SECRET", "dev-secret-change-me-please-use-32-bytes-minimum")
        )
    )
    admin_username: str = field(default_factory=lambda: os.getenv("ADMIN_USERNAME", "admin"))
    admin_password: str = field(default_factory=lambda: os.getenv("ADMIN_PASSWORD", ""))
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            x.strip()
            for x in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
            if x.strip()
        )
    )
    seed_demo: bool = field(default_factory=lambda: _bool("SEED_DEMO", False))
    demo_csv: str = field(
        default_factory=lambda: os.getenv("DEMO_CSV", str(REPO_ROOT / "data" / "crawl" / "weibo_1000.csv"))
    )
    local_model_path: str = field(
        default_factory=lambda: os.getenv(
            "LOCAL_MODEL_PATH", str(REPO_ROOT / "output" / "models" / "baselines.joblib")
        )
    )
    crawler_mode: str = field(default_factory=lambda: os.getenv("CRAWLER_MODE", "mock").lower())
    crawl_rate_limit_seconds: float = field(
        default_factory=lambda: _float("CRAWL_RATE_LIMIT_SECONDS", 2.5)
    )
    crawl_max_pages: int = field(default_factory=lambda: _int("CRAWL_MAX_PAGES", 50))
    crawl_repost_threshold: int = field(default_factory=lambda: _int("CRAWL_REPOST_THRESHOLD", 20))
    raw_retention_days: int = field(default_factory=lambda: _int("RAW_RETENTION_DAYS", 90))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-3-5-haiku-latest"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def ensure_dirs(self) -> None:
        (REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / "data" / "qrcodes").mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()


def random_password(length: int = 16) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%"
    return "".join(secrets.choice(alphabet) for _ in range(length))
