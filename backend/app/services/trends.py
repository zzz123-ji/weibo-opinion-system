"""趋势聚合：声量、情感占比、负面突增、高互动负面帖。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db import engine


def _bucket_sql(granularity: str) -> str:
    if engine.dialect.name == "postgresql":
        if granularity == "hour":
            return "to_char(date_trunc('hour', p.created_at), 'YYYY-MM-DD HH24:00')"
        return "to_char(date_trunc('day', p.created_at), 'YYYY-MM-DD')"
    if granularity == "hour":
        return "strftime('%Y-%m-%d %H:00', p.created_at)"
    return "strftime('%Y-%m-%d', p.created_at)"


def _time_where(time_from: datetime | None, time_to: datetime | None) -> tuple[str, dict]:
    clauses: list[str] = []
    params: dict = {}
    if time_from:
        clauses.append("p.created_at >= :time_from")
        params["time_from"] = time_from
    if time_to:
        clauses.append("p.created_at <= :time_to")
        params["time_to"] = time_to
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def _keyword_join(keywords: list[str] | None) -> tuple[str, dict]:
    if not keywords:
        return "", {}
    placeholders = ", ".join(f":kw{i}" for i in range(len(keywords)))
    sql = (
        " JOIN post_source ps ON ps.mid = p.mid "
        " AND ps.source_type = 'keyword' "
        f" AND ps.source_value IN ({placeholders}) "
    )
    params = {f"kw{i}": kw for i, kw in enumerate(keywords)}
    return sql, params


def volume_series(
    db: Session,
    granularity: str = "day",
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    keywords: list[str] | None = None,
    split_by_keyword: bool = False,
) -> dict:
    granularity = "hour" if granularity == "hour" else "day"
    bucket = _bucket_sql(granularity)
    join_sql, kw_params = _keyword_join(keywords)
    where_sql, time_params = _time_where(time_from, time_to)
    params = {**kw_params, **time_params}
    if split_by_keyword and keywords:
        sql = f"""
            SELECT {bucket} AS bucket, ps.source_value AS series, COUNT(*) AS n
            FROM post p
            {join_sql}
            WHERE 1=1 {where_sql}
            GROUP BY bucket, ps.source_value
            ORDER BY bucket, ps.source_value
        """
        rows = db.execute(text(sql), params).mappings().all()
        series: dict[str, list[dict]] = defaultdict(list)
        total: dict[str, int] = defaultdict(int)
        for row in rows:
            series[row["series"]].append({"bucket": row["bucket"], "count": int(row["n"])})
            total[row["bucket"]] += int(row["n"])
        return {
            "granularity": granularity,
            "points": [{"bucket": k, "count": v} for k, v in sorted(total.items())],
            "series": dict(series),
        }

    sql = f"""
        SELECT {bucket} AS bucket, COUNT(*) AS n
        FROM post p
        {join_sql}
        WHERE 1=1 {where_sql}
        GROUP BY bucket
        ORDER BY bucket
    """
    rows = db.execute(text(sql), params).mappings().all()
    return {
        "granularity": granularity,
        "points": [{"bucket": row["bucket"], "count": int(row["n"])} for row in rows],
        "series": {},
    }


def sentiment_series(
    db: Session,
    granularity: str = "day",
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    keywords: list[str] | None = None,
) -> dict:
    granularity = "hour" if granularity == "hour" else "day"
    bucket = _bucket_sql(granularity)
    join_sql, kw_params = _keyword_join(keywords)
    where_sql, time_params = _time_where(time_from, time_to)
    params = {**kw_params, **time_params}
    sql = f"""
        SELECT {bucket} AS bucket,
               SUM(CASE WHEN f.label = 'pos' THEN 1 ELSE 0 END) AS pos,
               SUM(CASE WHEN f.label = 'neu' THEN 1 ELSE 0 END) AS neu,
               SUM(CASE WHEN f.label = 'neg' THEN 1 ELSE 0 END) AS neg,
               SUM(CASE WHEN f.label IS NULL OR f.source = 'unscored' THEN 1 ELSE 0 END) AS unscored,
               COUNT(*) AS total
        FROM post p
        {join_sql}
        LEFT JOIN v_sentiment_final f
               ON f.text_type = 'post' AND f.text_id = p.mid
        WHERE 1=1 {where_sql}
        GROUP BY bucket
        ORDER BY bucket
    """
    rows = db.execute(text(sql), params).mappings().all()
    points = []
    for row in rows:
        total = max(1, int(row["total"]))
        points.append({
            "bucket": row["bucket"],
            "pos": int(row["pos"] or 0),
            "neu": int(row["neu"] or 0),
            "neg": int(row["neg"] or 0),
            "unscored": int(row["unscored"] or 0),
            "pos_ratio": round(int(row["pos"] or 0) / total, 6),
            "neu_ratio": round(int(row["neu"] or 0) / total, 6),
            "neg_ratio": round(int(row["neg"] or 0) / total, 6),
            "unscored_ratio": round(int(row["unscored"] or 0) / total, 6),
        })
    return {"granularity": granularity, "points": points}


def negative_alerts(
    db: Session,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    threshold_ratio: float = 3.0,
    min_count: int = 10,
    lookback_days: int = 7,
) -> list[dict]:
    """按小时检测负面突增：当前小时 >= 前 7 天同时段均值 * ratio，且 >= min_count。"""
    now = datetime.utcnow()
    time_to = time_to or now
    time_from = time_from or (time_to - timedelta(days=7))
    start = time_from - timedelta(days=lookback_days)
    sql = """
        SELECT strftime('%Y-%m-%d %H:00', p.created_at) AS bucket,
               COUNT(*) AS n
        FROM post p
        JOIN v_sentiment_final f
          ON f.text_type = 'post' AND f.text_id = p.mid
        WHERE f.label = 'neg'
          AND p.created_at >= :start
          AND p.created_at <= :end
        GROUP BY bucket
        ORDER BY bucket
    """ if engine.dialect.name != "postgresql" else """
        SELECT to_char(date_trunc('hour', p.created_at), 'YYYY-MM-DD HH24:00') AS bucket,
               COUNT(*) AS n
        FROM post p
        JOIN v_sentiment_final f
          ON f.text_type = 'post' AND f.text_id = p.mid
        WHERE f.label = 'neg'
          AND p.created_at >= :start
          AND p.created_at <= :end
        GROUP BY bucket
        ORDER BY bucket
    """
    rows = db.execute(text(sql), {"start": start, "end": time_to}).mappings().all()
    counts = {row["bucket"]: int(row["n"]) for row in rows}

    # 以“小时”为键建立历史桶；同一小时-of-day 的前 7 天均值作为基线。
    by_key = {}
    for bucket, n in counts.items():
        dt = datetime.strptime(bucket, "%Y-%m-%d %H:00")
        by_key[dt] = n

    alerts = []
    cursor = time_from.replace(minute=0, second=0, microsecond=0)
    end = time_to.replace(minute=0, second=0, microsecond=0)
    while cursor <= end:
        current = by_key.get(cursor, 0)
        history = []
        for d in range(1, lookback_days + 1):
            history.append(by_key.get(cursor - timedelta(days=d), 0))
        baseline = sum(history) / len(history) if history else 0.0
        ratio = (current / baseline) if baseline > 0 else (999.0 if current > 0 else 0.0)
        if current >= min_count and ratio >= threshold_ratio:
            alerts.append({
                "bucket": cursor.strftime("%Y-%m-%d %H:00"),
                "neg_count": current,
                "baseline": round(baseline, 2),
                "ratio": round(ratio, 2),
                "threshold_ratio": threshold_ratio,
                "min_count": min_count,
            })
        cursor += timedelta(hours=1)
    return alerts


def top_negative(
    db: Session,
    limit: int = 10,
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    keywords: list[str] | None = None,
) -> list[dict]:
    join_sql, kw_params = _keyword_join(keywords)
    where_sql, time_params = _time_where(time_from, time_to)
    params = {**kw_params, **time_params, "limit": limit}
    sql = f"""
        SELECT p.mid, p.text, p.created_at,
               p.reposts + p.comments + p.likes AS interactions,
               a.nickname AS author_nickname,
               f.label, f.source
        FROM post p
        {join_sql}
        LEFT JOIN author a ON a.uid = p.author_uid
        JOIN v_sentiment_final f
          ON f.text_type = 'post' AND f.text_id = p.mid
        WHERE f.label = 'neg' {where_sql}
        ORDER BY interactions DESC, p.created_at DESC
        LIMIT :limit
    """
    rows = db.execute(text(sql), params).mappings().all()
    return [
        {
            "mid": row["mid"],
            "text": row["text"],
            "author_nickname": row["author_nickname"] or "",
            "created_at": row["created_at"],
            "interactions": int(row["interactions"] or 0),
            "sentiment": row["label"],
            "sentiment_source": row["source"] or "unscored",
        }
        for row in rows
    ]
