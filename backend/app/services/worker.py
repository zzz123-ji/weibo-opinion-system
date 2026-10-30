"""情感 worker：从 sentiment_queue 取任务，调用引擎，写 sentiment_auto。"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from sqlalchemy.orm import Session

from ..db import engine as db_engine
from ..models import SentimentAuto, SentimentQueue, Setting
from .final import get_text
from .sentiment import build_engine

logger = logging.getLogger(__name__)


def active_engine(db: Session) -> str:
    row = db.get(Setting, "active_engine")
    value = row.value if row else "local"
    if isinstance(value, str):
        return value.strip('"')
    return "local"


def claim_batch(db: Session, limit: int = 64) -> list[SentimentQueue]:
    q = (
        db.query(SentimentQueue)
        .filter(SentimentQueue.status == "pending")
        .order_by(SentimentQueue.id.asc())
    )
    if db_engine.dialect.name == "postgresql":
        q = q.with_for_update(skip_locked=True)
    rows = q.limit(limit).all()
    now = datetime.utcnow()
    for row in rows:
        row.status = "processing"
        row.locked_at = now
        row.attempts += 1
    db.commit()
    return rows


def process_batch(
    db: Session,
    engine_name: str | None = None,
    version: str | None = None,
    limit: int = 64,
) -> int:
    rows = claim_batch(db, limit)
    if not rows:
        return 0
    engine_name = engine_name or active_engine(db)
    try:
        engine = build_engine(engine_name, version)
    except Exception as exc:
        for row in rows:
            row.status = "failed"
            row.last_error = str(exc)
        db.commit()
        logger.error("引擎初始化失败：%s", exc)
        return 0

    valid: list[tuple[SentimentQueue, str]] = []
    for row in rows:
        info = get_text(db, row.text_type, row.text_id)
        if not info:
            row.status = "failed"
            row.last_error = "文本不存在"
            continue
        valid.append((row, info[0]))
    if not valid:
        db.commit()
        return 0

    try:
        results = engine.classify([text for _, text in valid])
    except Exception as exc:
        for row, _ in valid:
            row.status = "pending" if row.attempts < 3 else "failed"
            row.last_error = str(exc)
            row.locked_at = None
        db.commit()
        logger.warning("引擎批量失败，已退回队列：%s", exc)
        return 0

    for (row, _text), (label, score) in zip(valid, results):
        existing = (
            db.query(SentimentAuto)
            .filter(
                SentimentAuto.text_type == row.text_type,
                SentimentAuto.text_id == row.text_id,
                SentimentAuto.engine == engine_name,
                SentimentAuto.engine_version == engine.version,
            )
            .first()
        )
        if existing:
            existing.label = label
            existing.score = score
            existing.created_at = datetime.utcnow()
        else:
            db.add(
                SentimentAuto(
                    text_type=row.text_type,
                    text_id=row.text_id,
                    engine=engine_name,
                    engine_version=engine.version,
                    label=label,
                    score=score,
                )
            )
        row.status = "done"
        row.last_error = None
        row.locked_at = None
    db.commit()
    return len(valid)


def process_queue(
    db: Session,
    engine_name: str | None = None,
    version: str | None = None,
    batch_size: int = 64,
    max_batches: int = 100,
    sleep_seconds: float = 0.0,
) -> int:
    total = 0
    for _ in range(max_batches):
        n = process_batch(db, engine_name, version, batch_size)
        if n == 0:
            break
        total += n
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return total
