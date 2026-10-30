from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="weibo-test-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["ADMIN_PASSWORD"] = "admin123"
os.environ["JWT_SECRET"] = "test-secret-" + "x" * 48
os.environ["SEED_DEMO"] = "0"
os.environ["CRAWLER_MODE"] = "mock"

from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, SessionLocal, engine, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models import (  # noqa: E402
    AppUser,
    Author,
    Keyword,
    Post,
    PostMetric,
    PostSource,
    SentimentAuto,
    Setting,
)
from app.security import hash_password  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(bind=engine)
    init_db()
    db = SessionLocal()
    try:
        db.add(AppUser(username="admin", password_hash=hash_password("admin123"), role="admin"))
        db.add(
            AppUser(
                username="annotator",
                password_hash=hash_password("annotator123"),
                role="annotator",
            )
        )
        db.merge(Setting(key="active_engine", value="mock"))
        db.merge(Setting(key="active_engine_version", value="test"))
        db.commit()
    finally:
        db.close()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def login(client: TestClient, username: str = "admin", password: str = "admin123") -> dict:
    resp = client.post("/api/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture()
def admin_headers(client):
    return login(client)


@pytest.fixture()
def annotator_headers(client):
    return login(client, "annotator", "annotator123")


def add_post(
    mid: str,
    text: str,
    created_at,
    *,
    label: str = "neg",
    score: float = 0.8,
    reposts: int = 10,
    comments: int = 2,
    likes: int = 5,
    source: str = "武汉",
):
    db = SessionLocal()
    try:
        uid = "u_test"
        if not db.get(Author, uid):
            db.add(Author(uid=uid, nickname="测试作者", followers=100))
        db.add(
            Post(
                mid=mid,
                text=text,
                author_uid=uid,
                created_at=created_at,
                reposts=reposts,
                comments=comments,
                likes=likes,
                raw={"seed": True},
            )
        )
        db.add(PostMetric(mid=mid, ts=created_at, reposts=reposts, comments=comments, likes=likes))
        db.add(
            PostSource(
                mid=mid,
                source_type="keyword",
                source_value=source,
                first_seen_at=created_at,
            )
        )
        db.add(
            SentimentAuto(
                text_type="post",
                text_id=mid,
                engine="mock",
                engine_version="test",
                label=label,
                score=score,
            )
        )
        db.commit()
    finally:
        db.close()
