"""m.weibo.cn / weibo.com JSON 解析。

解析层只做“原始 JSON -> 标准化 dict”，不访问数据库，便于用录制样本做单测。
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone
from typing import Any


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()


def parse_created_at(value: Any, now: datetime | None = None) -> datetime:
    now = now or datetime.utcnow()
    if value is None or value == "":
        return now
    if isinstance(value, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(value))
        except (ValueError, OSError):
            return now
    text = str(value).strip()
    if text.isdigit():
        try:
            return datetime.utcfromtimestamp(float(text))
        except (ValueError, OSError):
            return now
    if text in {"刚刚", "现在"}:
        return now
    m = re.match(r"(\d+)\s*秒前", text)
    if m:
        return now - timedelta(seconds=int(m.group(1)))
    m = re.match(r"(\d+)\s*分钟前", text)
    if m:
        return now - timedelta(minutes=int(m.group(1)))
    m = re.match(r"(\d+)\s*小时前", text)
    if m:
        return now - timedelta(hours=int(m.group(1)))
    if text == "昨天":
        return now - timedelta(days=1)
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%m-%d"):
        try:
            dt = datetime.strptime(text, fmt)
            if fmt == "%m-%d":
                dt = dt.replace(year=now.year)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except ValueError:
            pass
    return now


def parse_author(user: dict | None) -> dict:
    user = user or {}
    uid = str(user.get("id") or user.get("idstr") or user.get("uid") or "")
    return {
        "uid": uid,
        "nickname": clean_text(user.get("screen_name") or user.get("name") or ""),
        "followers": int(user.get("followers_count") or user.get("followers") or 0),
        "verified": bool(user.get("verified") or user.get("verified_type") not in (None, -1)),
    }


def parse_mblog(mblog: dict, source_type: str = "", source_value: str = "") -> dict | None:
    if not mblog:
        return None
    mid = str(mblog.get("mid") or mblog.get("id") or mblog.get("idstr") or "")
    if not mid:
        return None
    retweeted = mblog.get("retweeted_status") or {}
    root_mid = str(retweeted.get("mid") or retweeted.get("id") or "") if retweeted else None
    long_text = mblog.get("longText")
    if isinstance(long_text, dict):
        long_text = long_text.get("longTextContent")
    return {
        "mid": mid,
        "text": clean_text(mblog.get("text") or long_text or ""),
        "author": parse_author(mblog.get("user")),
        "created_at": parse_created_at(mblog.get("created_at")),
        "reposts": int(mblog.get("reposts_count") or 0),
        "comments": int(mblog.get("comments_count") or 0),
        "likes": int(mblog.get("attitudes_count") or 0),
        "is_repost": bool(retweeted),
        "root_mid": root_mid or None,
        "raw": mblog,
        "source_type": source_type,
        "source_value": source_value,
    }


def parse_cards(payload: dict, source_type: str = "", source_value: str = "") -> list[dict]:
    data = payload.get("data") or {}
    cards = data.get("cards") or []
    posts: list[dict] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        mblog = card.get("mblog")
        if not mblog and isinstance(card.get("card_group"), list):
            for sub in card["card_group"]:
                if isinstance(sub, dict) and sub.get("mblog"):
                    item = parse_mblog(sub["mblog"], source_type, source_value)
                    if item:
                        posts.append(item)
            continue
        if mblog:
            item = parse_mblog(mblog, source_type, source_value)
            if item:
                posts.append(item)
    return posts


def parse_search(payload: dict, keyword: str) -> list[dict]:
    return parse_cards(payload, "keyword", keyword)


def parse_user_timeline(payload: dict, uid: str) -> list[dict]:
    return parse_cards(payload, "account", uid)


def parse_comments(payload: dict, mid: str) -> list[dict]:
    data = payload.get("data") or {}
    rows = data.get("data") or data.get("comments") or []
    comments: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("id") or row.get("idstr") or "")
        if not cid:
            continue
        comments.append({
            "id": cid,
            "mid": mid,
            "text": clean_text(row.get("text") or ""),
            "author": parse_author(row.get("user")),
            "created_at": parse_created_at(row.get("created_at")),
            "likes": int(row.get("like_count") or row.get("likes_count") or 0),
            "reply_to": str(row.get("reply_id") or "") or None,
            "raw": row,
        })
    return comments


def parse_reposts(payload: dict, root_mid: str) -> list[dict]:
    data = payload.get("data") or {}
    rows = data.get("data") or data.get("reposts") or []
    reposts: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        mid = str(row.get("mid") or row.get("id") or row.get("idstr") or "")
        if not mid:
            continue
        reposts.append({
            "mid": mid,
            "root_mid": root_mid,
            "text": clean_text(row.get("text") or ""),
            "author": parse_author(row.get("user")),
            "created_at": parse_created_at(row.get("created_at")),
            "raw": row,
        })
    return reposts


def parse_hotsearch(payload: dict) -> list[dict]:
    data = payload.get("data") or {}
    rows = data.get("realtime") or data.get("hotsearch") or []
    result: list[dict] = []
    for rank, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        word = clean_text(row.get("word") or row.get("note") or "")
        if not word:
            continue
        result.append({
            "rank": rank,
            "word": word,
            "heat": int(row.get("num") or row.get("heat") or 0),
            "label": clean_text(row.get("label_name") or row.get("label") or ""),
        })
    return result
