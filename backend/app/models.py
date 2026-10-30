"""SQLAlchemy 数据模型，对应设计文档第 4 节。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base

JSONType = JSON().with_variant(JSONB, "postgresql")


class Keyword(Base):
    __tablename__ = "keyword"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    word: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    is_temp: Mapped[bool] = mapped_column(Boolean, default=False)
    expire_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class WatchAccount(Base):
    __tablename__ = "watch_account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uid: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    nickname: Mapped[str] = mapped_column(String(120), default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class WeiboCookie(Base):
    __tablename__ = "weibo_cookie"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_name: Mapped[str] = mapped_column(String(120), unique=True)
    cookie_enc: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="ok", index=True)
    cooling_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    fail_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Author(Base):
    __tablename__ = "author"

    uid: Mapped[str] = mapped_column(String(64), primary_key=True)
    nickname: Mapped[str] = mapped_column(String(120), default="")
    followers: Mapped[int] = mapped_column(Integer, default=0)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Post(Base):
    __tablename__ = "post"

    mid: Mapped[str] = mapped_column(String(64), primary_key=True)
    text: Mapped[str] = mapped_column(Text, default="")
    author_uid: Mapped[str | None] = mapped_column(ForeignKey("author.uid"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    reposts: Mapped[int] = mapped_column(Integer, default=0)
    comments: Mapped[int] = mapped_column(Integer, default=0)
    likes: Mapped[int] = mapped_column(Integer, default=0)
    is_repost: Mapped[bool] = mapped_column(Boolean, default=False)
    root_mid: Mapped[str | None] = mapped_column(String(64), nullable=True)
    raw: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_post_created_interactions", "created_at", "reposts", "comments", "likes"),)


class PostSource(Base):
    __tablename__ = "post_source"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mid: Mapped[str] = mapped_column(ForeignKey("post.mid", ondelete="CASCADE"), index=True)
    source_type: Mapped[str] = mapped_column(String(20), index=True)  # keyword / account / hotsearch
    source_value: Mapped[str] = mapped_column(String(200), default="")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("mid", "source_type", "source_value", name="uq_post_source"),
    )


class PostMetric(Base):
    __tablename__ = "post_metric"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mid: Mapped[str] = mapped_column(ForeignKey("post.mid", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    reposts: Mapped[int] = mapped_column(Integer, default=0)
    comments: Mapped[int] = mapped_column(Integer, default=0)
    likes: Mapped[int] = mapped_column(Integer, default=0)


class Comment(Base):
    __tablename__ = "comment"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    mid: Mapped[str] = mapped_column(ForeignKey("post.mid", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(Text, default="")
    author_uid: Mapped[str | None] = mapped_column(ForeignKey("author.uid"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    likes: Mapped[int] = mapped_column(Integer, default=0)
    reply_to: Mapped[str | None] = mapped_column(String(64), nullable=True)
    raw: Mapped[dict | None] = mapped_column(JSONType, nullable=True)


class Repost(Base):
    __tablename__ = "repost"

    mid: Mapped[str] = mapped_column(String(64), primary_key=True)
    root_mid: Mapped[str] = mapped_column(ForeignKey("post.mid", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(Text, default="")
    author_uid: Mapped[str | None] = mapped_column(ForeignKey("author.uid"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    raw: Mapped[dict | None] = mapped_column(JSONType, nullable=True)


class HotSearchSnapshot(Base):
    __tablename__ = "hot_search_snapshot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    rank: Mapped[int] = mapped_column(Integer)
    word: Mapped[str] = mapped_column(String(200), index=True)
    heat: Mapped[int] = mapped_column(Integer, default=0)
    label: Mapped[str] = mapped_column(String(40), default="")
    matched_keyword_id: Mapped[int | None] = mapped_column(ForeignKey("keyword.id"), nullable=True)

    __table_args__ = (UniqueConstraint("ts", "rank", name="uq_hotsearch_ts_rank"),)


class SentimentQueue(Base):
    __tablename__ = "sentiment_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text_type: Mapped[str] = mapped_column(String(20))
    text_id: Mapped[str] = mapped_column(String(64))
    enqueued_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (UniqueConstraint("text_type", "text_id", name="uq_sentiment_queue_text"),)


class SentimentAuto(Base):
    __tablename__ = "sentiment_auto"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text_type: Mapped[str] = mapped_column(String(20), index=True)
    text_id: Mapped[str] = mapped_column(String(64), index=True)
    engine: Mapped[str] = mapped_column(String(40), index=True)
    engine_version: Mapped[str] = mapped_column(String(60), default="")
    label: Mapped[str] = mapped_column(String(10), index=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        UniqueConstraint("text_type", "text_id", "engine", "engine_version", name="uq_sentiment_auto"),
    )


class SentimentLabel(Base):
    __tablename__ = "sentiment_label"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text_type: Mapped[str] = mapped_column(String(20), index=True)
    text_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("app_user.id"), index=True)
    label: Mapped[str] = mapped_column(String(10), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("text_type", "text_id", "user_id", name="uq_sentiment_label_user"),
    )


class AppUser(Base):
    __tablename__ = "app_user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="annotator")  # admin / annotator
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CrawlLog(Base):
    __tablename__ = "crawl_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_type: Mapped[str] = mapped_column(String(40), index=True)
    target: Mapped[str] = mapped_column(String(255), default="")
    cookie_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="running")
    items: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    error_msg: Mapped[str | None] = mapped_column(Text, nullable=True)


class Setting(Base):
    __tablename__ = "setting"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[dict | str | None] = mapped_column(JSONType, nullable=True)
