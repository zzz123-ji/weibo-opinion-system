"""待打分队列。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..models import Comment, Post, Repost, SentimentQueue


def enqueue_text(db: Session, text_type: str, text_id: str) -> None:
    row = (
        db.query(SentimentQueue)
        .filter(SentimentQueue.text_type == text_type, SentimentQueue.text_id == text_id)
        .first()
    )
    if row:
        if row.status in {"done", "failed"}:
            row.status = "pending"
            row.attempts = 0
            row.last_error = None
            row.locked_at = None
            row.enqueued_at = datetime.utcnow()
        return
    db.add(SentimentQueue(text_type=text_type, text_id=text_id))


def enqueue_all(
    db: Session,
    text_type: str | None = None,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    limit: int = 1000,
) -> int:
    types = [text_type] if text_type else ["post", "comment", "repost"]
    total = 0
    for tt in types:
        if tt == "post":
            q = db.query(Post.mid, Post.created_at)
        elif tt == "comment":
            q = db.query(Comment.id, Comment.created_at)
        elif tt == "repost":
            q = db.query(Repost.mid, Repost.created_at)
        else:
            continue
        if time_from:
            q = q.filter((Post.created_at if tt == "post" else Comment.created_at if tt == "comment" else Repost.created_at) >= time_from)
        if time_to:
            q = q.filter((Post.created_at if tt == "post" else Comment.created_at if tt == "comment" else Repost.created_at) <= time_to)
        rows = q.limit(limit).all()
        for text_id, _created in rows:
            enqueue_text(db, tt, str(text_id))
            total += 1
    db.commit()
    return total


def queue_counts(db: Session) -> dict[str, int]:
    rows = db.execute(
        text("SELECT status, COUNT(*) AS n FROM sentiment_queue GROUP BY status")
    ).mappings().all()
    return {row["status"]: int(row["n"]) for row in rows}
