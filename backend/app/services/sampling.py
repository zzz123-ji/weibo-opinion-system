"""标注抽样：随机、低置信度优先、双引擎不一致优先。"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..models import Comment, Post, Repost, SentimentAuto, SentimentLabel, Setting


def _candidate_rows(
    db: Session,
    text_type: str | None,
    time_from: datetime | None,
    time_to: datetime | None,
    keyword: str | None,
    limit: int = 3000,
) -> list[dict]:
    types = [text_type] if text_type else ["post", "comment", "repost"]
    rows: list[dict] = []
    for tt in types:
        if tt == "post":
            sql = """
                SELECT p.mid AS text_id, p.text, p.created_at, NULL AS root_mid
                FROM post p
                WHERE 1=1
            """
            params: dict = {}
            if time_from:
                sql += " AND p.created_at >= :time_from"
                params["time_from"] = time_from
            if time_to:
                sql += " AND p.created_at <= :time_to"
                params["time_to"] = time_to
            if keyword:
                sql += (
                    " AND EXISTS (SELECT 1 FROM post_source ps WHERE ps.mid = p.mid"
                    " AND ps.source_type = 'keyword' AND ps.source_value = :keyword)"
                )
                params["keyword"] = keyword
            sql += " ORDER BY p.created_at DESC LIMIT :limit"
            params["limit"] = limit
            for r in db.execute(text(sql), params).mappings().all():
                rows.append({"text_type": "post", **dict(r)})
        elif tt == "comment":
            sql = """
                SELECT c.id AS text_id, c.text, c.created_at, c.mid AS root_mid
                FROM comment c JOIN post p ON p.mid = c.mid
                WHERE 1=1
            """
            params = {}
            if time_from:
                sql += " AND c.created_at >= :time_from"
                params["time_from"] = time_from
            if time_to:
                sql += " AND c.created_at <= :time_to"
                params["time_to"] = time_to
            if keyword:
                sql += (
                    " AND EXISTS (SELECT 1 FROM post_source ps WHERE ps.mid = p.mid"
                    " AND ps.source_type = 'keyword' AND ps.source_value = :keyword)"
                )
                params["keyword"] = keyword
            sql += " ORDER BY c.created_at DESC LIMIT :limit"
            params["limit"] = limit
            for r in db.execute(text(sql), params).mappings().all():
                rows.append({"text_type": "comment", **dict(r)})
        elif tt == "repost":
            sql = """
                SELECT r.mid AS text_id, r.text, r.created_at, r.root_mid
                FROM repost r JOIN post p ON p.mid = r.root_mid
                WHERE 1=1
            """
            params = {}
            if time_from:
                sql += " AND r.created_at >= :time_from"
                params["time_from"] = time_from
            if time_to:
                sql += " AND r.created_at <= :time_to"
                params["time_to"] = time_to
            if keyword:
                sql += (
                    " AND EXISTS (SELECT 1 FROM post_source ps WHERE ps.mid = p.mid"
                    " AND ps.source_type = 'keyword' AND ps.source_value = :keyword)"
                )
                params["keyword"] = keyword
            sql += " ORDER BY r.created_at DESC LIMIT :limit"
            params["limit"] = limit
            for r in db.execute(text(sql), params).mappings().all():
                rows.append({"text_type": "repost", **dict(r)})
    return rows


def _auto_labels(db: Session, text_type: str, text_id: str) -> list[SentimentAuto]:
    return (
        db.query(SentimentAuto)
        .filter(SentimentAuto.text_type == text_type, SentimentAuto.text_id == text_id)
        .order_by(SentimentAuto.created_at.desc())
        .all()
    )


def _active_engine(db: Session) -> str:
    setting = db.get(Setting, "active_engine")
    value = setting.value if setting else "local"
    if isinstance(value, str):
        return value.strip('"')
    return "local"


def next_sample(
    db: Session,
    user_id: int,
    strategy: str = "random",
    text_type: str | None = None,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    keyword: str | None = None,
) -> dict | None:
    candidates = _candidate_rows(db, text_type, time_from, time_to, keyword)
    if not candidates:
        return None

    labeled = {
        (r.text_type, r.text_id)
        for r in db.query(SentimentLabel)
        .filter(SentimentLabel.user_id == user_id)
        .all()
    }
    candidates = [c for c in candidates if (c["text_type"], c["text_id"]) not in labeled]
    if not candidates:
        return None

    active = _active_engine(db)
    auto_cache: dict[tuple[str, str], list[SentimentAuto]] = {}
    for c in candidates:
        key = (c["text_type"], c["text_id"])
        auto_cache[key] = _auto_labels(db, *key)

    if strategy == "low_confidence":
        def low_score(c: dict) -> float:
            rows = [r for r in auto_cache[(c["text_type"], c["text_id"])] if r.engine == active]
            return min((r.score for r in rows), default=1.0)

        candidates.sort(key=low_score)
    elif strategy == "disagreement":
        def disagreement(c: dict) -> int:
            labels = {r.label for r in auto_cache[(c["text_type"], c["text_id"])]}
            return len(labels)

        candidates.sort(key=disagreement, reverse=True)
    else:
        random.shuffle(candidates)

    chosen = candidates[0]
    context = None
    if chosen["root_mid"]:
        post = db.query(Post).filter(Post.mid == chosen["root_mid"]).first()
        if post:
            context = {"mid": post.mid, "text": post.text, "created_at": post.created_at.isoformat()}
    return {
        "text_type": chosen["text_type"],
        "text_id": chosen["text_id"],
        "text": chosen["text"],
        "post_context": context,
        "auto_labels": auto_cache[(chosen["text_type"], chosen["text_id"])],
    }


def label_stats(db: Session, user_id: int) -> dict:
    from datetime import date

    total_texts = (
        db.query(Post).count() + db.query(Comment).count() + db.query(Repost).count()
    )
    labeled_texts = db.execute(
        text("SELECT COUNT(DISTINCT text_type || ':' || text_id) AS n FROM sentiment_label")
    ).scalar() or 0
    my_total = db.query(SentimentLabel).filter(SentimentLabel.user_id == user_id).count()
    today_start = datetime.combine(date.today(), datetime.min.time())
    my_today = (
        db.query(SentimentLabel)
        .filter(SentimentLabel.user_id == user_id, SentimentLabel.created_at >= today_start)
        .count()
    )
    annotators = db.execute(
        text("SELECT COUNT(DISTINCT user_id) AS n FROM sentiment_label")
    ).scalar() or 0

    # 两两一致率：对每个至少两人标注的文本，计算一致对 / 总对。
    rows = db.execute(
        text(
            "SELECT text_type, text_id, label, user_id FROM sentiment_label "
            "ORDER BY text_type, text_id, user_id"
        )
    ).mappings().all()
    grouped: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        grouped.setdefault((row["text_type"], row["text_id"]), []).append(row["label"])
    agree = 0
    pairs = 0
    for labels in grouped.values():
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pairs += 1
                if labels[i] == labels[j]:
                    agree += 1
    return {
        "total_texts": total_texts,
        "labeled_texts": int(labeled_texts),
        "my_total": my_total,
        "my_today": my_today,
        "annotators": int(annotators),
        "pairwise_agreement": round(agree / pairs, 4) if pairs else None,
        "pairwise_pairs": pairs,
    }
