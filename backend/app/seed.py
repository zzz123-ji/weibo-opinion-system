"""初始化管理员与演示数据。

演示数据来自项目已有的 1,000 条公开数据落盘样本，只用于把系统跑通；
不伪造人工标注，也不冒充实时微博。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

from .config import REPO_ROOT, random_password, settings
from .db import Base, SessionLocal, init_db
from .models import (
    AppUser,
    Author,
    HotSearchSnapshot,
    Keyword,
    Post,
    PostMetric,
    PostSource,
    SentimentAuto,
    Setting,
)
from .security import hash_password
from .services.sentiment import build_engine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEMO_KEYWORDS = ["武汉", "口罩", "医院", "疫情", "加油"]


def parse_time(value: str | None) -> datetime:
    if not value:
        return datetime(2020, 1, 1)
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return datetime(2020, 1, 1)


def deterministic_int(seed: str, low: int, high: int) -> int:
    h = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    return low + h % (high - low + 1)


def ensure_admin(db: Session) -> AppUser:
    admin = db.query(AppUser).filter(AppUser.username == settings.admin_username).first()
    if admin:
        return admin
    password = settings.admin_password or random_password()
    admin = AppUser(
        username=settings.admin_username,
        password_hash=hash_password(password),
        role="admin",
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)
    if not settings.admin_password:
        logger.warning("=" * 64)
        logger.warning("首次启动已创建管理员：%s", admin.username)
        logger.warning("初始密码：%s", password)
        logger.warning("请登录后立即修改；也可通过 ADMIN_PASSWORD 环境变量固定初始密码。")
        logger.warning("=" * 64)
    return admin


def ensure_annotator(db: Session) -> None:
    if db.query(AppUser).filter(AppUser.username == "annotator").first():
        return
    db.add(
        AppUser(
            username="annotator",
            password_hash=hash_password("annotator123"),
            role="annotator",
        )
    )
    db.commit()
    logger.info("已创建演示标注员 annotator / annotator123")


def seed_demo(db: Session, csv_path: str | Path) -> int:
    path = Path(csv_path)
    if not path.exists():
        logger.warning("演示 CSV 不存在，跳过：%s", path)
        return 0
    if db.query(Post).count() > 0:
        logger.info("数据库已有帖子，跳过演示数据导入")
        return 0

    admin = ensure_admin(db)
    ensure_annotator(db)

    keywords: list[Keyword] = []
    for word in DEMO_KEYWORDS:
        row = db.query(Keyword).filter(Keyword.word == word).first()
        if not row:
            row = Keyword(word=word, created_by=admin.id)
            db.add(row)
            db.flush()
        keywords.append(row)

    authors: dict[str, Author] = {}
    posts: list[Post] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get("user_id") or "u_unknown"
            if uid not in authors:
                author = Author(
                    uid=uid,
                    nickname=f"用户{uid[-4:]}",
                    followers=deterministic_int(uid, 10, 50000),
                    verified=False,
                )
                authors[uid] = author
                db.add(author)
            mid = row.get("post_id") or hashlib.sha256(row.get("text", "").encode()).hexdigest()[:16]
            text = row.get("text") or ""
            created = parse_time(row.get("time"))
            reposts = deterministic_int(mid + "r", 0, 120)
            comments = deterministic_int(mid + "c", 0, 80)
            likes = deterministic_int(mid + "l", 0, 600)
            post = Post(
                mid=mid,
                text=text,
                author_uid=uid,
                created_at=created,
                reposts=reposts,
                comments=comments,
                likes=likes,
                is_repost=False,
                raw={"source": "demo_seed", "label": row.get("label")},
                fetched_at=datetime.utcnow(),
            )
            db.add(post)
            posts.append(post)
            source_word = next((w for w in DEMO_KEYWORDS if w in text), "疫情")
            db.add(
                PostSource(
                    mid=mid,
                    source_type="keyword",
                    source_value=source_word,
                    first_seen_at=created,
                )
            )
            db.add(
                PostMetric(
                    mid=mid,
                    ts=created,
                    reposts=reposts,
                    comments=comments,
                    likes=likes,
                )
            )
    db.commit()

    # 热搜快照：只做趋势展示，不代表实时榜单。
    base = datetime(2020, 1, 25)
    for day in range(10):
        ts = base + timedelta(days=day)
        for rank, word in enumerate(DEMO_KEYWORDS, start=1):
            db.add(
                HotSearchSnapshot(
                    ts=ts,
                    rank=rank,
                    word=word,
                    heat=100000 - rank * 10000 - day * 1000,
                    label="疫情",
                    matched_keyword_id=keywords[rank - 1].id,
                )
            )
    db.commit()

    # 用本地引擎生成自动分；不写人工标注。
    engine = build_engine("local")
    texts = [p.text for p in posts]
    labels = engine.classify(texts)
    for post, (label, score) in zip(posts, labels):
        db.add(
            SentimentAuto(
                text_type="post",
                text_id=post.mid,
                engine="local",
                engine_version=engine.version,
                label=label,
                score=score,
            )
        )
    db.merge(Setting(key="active_engine", value="local"))
    db.merge(Setting(key="active_engine_version", value=engine.version))
    db.commit()
    logger.info("演示数据导入完成：%s 条帖子，引擎 %s / %s", len(posts), engine.name, engine.version)
    return len(posts)


def main() -> int:
    ap = argparse.ArgumentParser(description="初始化管理员与演示数据")
    ap.add_argument("--csv", default=settings.demo_csv)
    ap.add_argument("--reset", action="store_true", help="删除并重建全部表（危险）")
    ap.add_argument("--admin-only", action="store_true")
    ap.add_argument("--if-empty", action="store_true")
    args = ap.parse_args()

    if args.reset:
        from .db import Base, engine

        Base.metadata.drop_all(bind=engine)
    init_db()
    db = SessionLocal()
    try:
        ensure_admin(db)
        if not args.admin_only:
            seed_demo(db, args.csv)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
