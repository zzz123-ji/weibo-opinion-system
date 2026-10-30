"""v_sentiment_final 查询与文本上下文。"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..models import Comment, Post, Repost


def get_final(db: Session, text_type: str, text_id: str) -> tuple[str | None, str]:
    row = db.execute(
        text(
            "SELECT label, source FROM v_sentiment_final "
            "WHERE text_type = :tt AND text_id = :tid"
        ),
        {"tt": text_type, "tid": text_id},
    ).mappings().first()
    if not row:
        return None, "unscored"
    return row["label"], row["source"]


def final_map(db: Session, text_type: str, ids: list[str]) -> dict[str, tuple[str | None, str]]:
    if not ids:
        return {}
    placeholders = ", ".join(f":id{i}" for i in range(len(ids)))
    params = {f"id{i}": v for i, v in enumerate(ids)}
    rows = db.execute(
        text(
            f"SELECT text_id, label, source FROM v_sentiment_final "
            f"WHERE text_type = :tt AND text_id IN ({placeholders})"
        ),
        {"tt": text_type, **params},
    ).mappings().all()
    return {row["text_id"]: (row["label"], row["source"]) for row in rows}


def get_text(db: Session, text_type: str, text_id: str) -> tuple[str, dict | None] | None:
    """返回 (正文, 原帖上下文)。评论/转发会带所属帖子。"""
    if text_type == "post":
        post = db.query(Post).filter(Post.mid == text_id).first()
        if not post:
            return None
        return post.text, None
    if text_type == "comment":
        comment = db.query(Comment).filter(Comment.id == text_id).first()
        if not comment:
            return None
        post = db.query(Post).filter(Post.mid == comment.mid).first()
        context = None
        if post:
            context = {"mid": post.mid, "text": post.text, "created_at": post.created_at.isoformat()}
        return comment.text, context
    if text_type == "repost":
        repost = db.query(Repost).filter(Repost.mid == text_id).first()
        if not repost:
            return None
        post = db.query(Post).filter(Post.mid == repost.root_mid).first()
        context = None
        if post:
            context = {"mid": post.mid, "text": post.text, "created_at": post.created_at.isoformat()}
        return repost.text, context
    return None
