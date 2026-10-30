"""数据库连接、建表与最终情感视图。"""

from __future__ import annotations

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


connect_args = {"check_same_thread": False} if settings.is_sqlite else {}
engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    future=True,
)

if settings.is_sqlite:

    @event.listens_for(engine, "connect")
    def _sqlite_pragma(dbapi_connection, _connection_record):  # pragma: no cover - 驱动级
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


FINAL_VIEW_SQL = """
CREATE VIEW {prefix}v_sentiment_final AS
WITH texts AS (
    SELECT 'post' AS text_type, mid AS text_id FROM post
    UNION ALL
    SELECT 'comment' AS text_type, id AS text_id FROM comment
    UNION ALL
    SELECT 'repost' AS text_type, mid AS text_id FROM repost
),
manual_votes AS (
    SELECT text_type, text_id, label, COUNT(*) AS votes, MAX(updated_at) AS latest
    FROM sentiment_label
    GROUP BY text_type, text_id, label
),
manual_majority AS (
    SELECT text_type, text_id, label
    FROM (
        SELECT text_type, text_id, label,
               ROW_NUMBER() OVER (
                   PARTITION BY text_type, text_id
                   ORDER BY votes DESC, latest DESC
               ) AS rn
        FROM manual_votes
    ) ranked
    WHERE rn = 1
),
auto_active AS (
    SELECT text_type, text_id, label
    FROM (
        SELECT sa.text_type, sa.text_id, sa.label,
               ROW_NUMBER() OVER (
                   PARTITION BY sa.text_type, sa.text_id
                   ORDER BY sa.created_at DESC, sa.id DESC
               ) AS rn
        FROM sentiment_auto sa
        JOIN setting st
          ON st.key = 'active_engine'
         AND sa.engine = REPLACE(CAST(st.value AS TEXT), '"', '')
    ) ranked_auto
    WHERE rn = 1
)
SELECT t.text_type,
       t.text_id,
       COALESCE(m.label, a.label) AS label,
       CASE
           WHEN m.label IS NOT NULL THEN 'manual'
           WHEN a.label IS NOT NULL THEN 'auto'
           ELSE 'unscored'
       END AS source
FROM texts t
LEFT JOIN manual_majority m
       ON m.text_type = t.text_type AND m.text_id = t.text_id
LEFT JOIN auto_active a
       ON a.text_type = t.text_type AND a.text_id = t.text_id
"""


def ensure_final_view() -> None:
    """创建 v_sentiment_final。

    SQLite 不支持 CREATE OR REPLACE VIEW，先 DROP 再建；Postgres 用 OR REPLACE。
    """
    dialect = engine.dialect.name
    with engine.begin() as conn:
        if dialect == "postgresql":
            conn.execute(text("DROP VIEW IF EXISTS v_sentiment_final"))
            conn.execute(text(FINAL_VIEW_SQL.format(prefix="")))
        else:
            conn.execute(text("DROP VIEW IF EXISTS v_sentiment_final"))
            conn.execute(text(FINAL_VIEW_SQL.format(prefix="")))


def init_db() -> None:
    from . import models  # noqa: F401  确保模型注册

    Base.metadata.create_all(bind=engine)
    ensure_final_view()
