"""舆情列表与详情。"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import Author, Comment, Post, PostMetric, PostSource, Repost
from ..schemas import (
    CommentOut,
    MetricPoint,
    PostDetail,
    PostListItem,
    PostPage,
    RepostOut,
    SourceOut,
)
from ..services.final import final_map, get_final

router = APIRouter(prefix="/posts", tags=["posts"], dependencies=[Depends(get_current_user)])


def _where(
    keyword: str | None,
    source: str | None,
    time_from: datetime | None,
    time_to: datetime | None,
    sentiment: str | None,
    min_interactions: int | None,
    q: str | None,
) -> tuple[str, dict]:
    clauses = ["1=1"]
    params: dict = {}
    if keyword:
        clauses.append(
            "EXISTS (SELECT 1 FROM post_source ps WHERE ps.mid = p.mid"
            " AND ps.source_type = 'keyword' AND ps.source_value = :keyword)"
        )
        params["keyword"] = keyword
    if source:
        clauses.append(
            "EXISTS (SELECT 1 FROM post_source ps WHERE ps.mid = p.mid"
            " AND ps.source_type = :source)"
        )
        params["source"] = source
    if time_from:
        clauses.append("p.created_at >= :time_from")
        params["time_from"] = time_from
    if time_to:
        clauses.append("p.created_at <= :time_to")
        params["time_to"] = time_to
    if sentiment:
        if sentiment == "unscored":
            clauses.append("(f.label IS NULL OR f.source = 'unscored')")
        else:
            clauses.append("f.label = :sentiment")
            params["sentiment"] = sentiment
    if min_interactions:
        clauses.append("(p.reposts + p.comments + p.likes) >= :min_interactions")
        params["min_interactions"] = min_interactions
    if q:
        clauses.append("p.text LIKE :q")
        params["q"] = f"%{q}%"
    return " AND ".join(clauses), params


@router.get("", response_model=PostPage)
def list_posts(
    keyword: str | None = None,
    source: str | None = Query(default=None, pattern="^(keyword|account|hotsearch)$"),
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    sentiment: str | None = Query(default=None, pattern="^(pos|neu|neg|unscored)$"),
    min_interactions: int | None = Query(default=None, ge=0),
    q: str | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="created_at_desc", pattern="^(created_at_desc|created_at_asc|interactions_desc)$"),
    db: Session = Depends(get_db),
):
    where, params = _where(keyword, source, time_from, time_to, sentiment, min_interactions, q)
    count_sql = f"""
        SELECT COUNT(*) AS n
        FROM post p
        LEFT JOIN v_sentiment_final f
               ON f.text_type = 'post' AND f.text_id = p.mid
        WHERE {where}
    """
    total = int(db.execute(text(count_sql), params).scalar() or 0)

    order = {
        "created_at_desc": "p.created_at DESC",
        "created_at_asc": "p.created_at ASC",
        "interactions_desc": "(p.reposts + p.comments + p.likes) DESC, p.created_at DESC",
    }[sort]
    params = {**params, "limit": page_size, "offset": (page - 1) * page_size}
    sql = f"""
        SELECT p.mid, p.text, p.author_uid, p.created_at, p.reposts, p.comments, p.likes,
               p.is_repost, a.nickname AS author_nickname,
               f.label AS sentiment, f.source AS sentiment_source
        FROM post p
        LEFT JOIN author a ON a.uid = p.author_uid
        LEFT JOIN v_sentiment_final f
               ON f.text_type = 'post' AND f.text_id = p.mid
        WHERE {where}
        ORDER BY {order}
        LIMIT :limit OFFSET :offset
    """
    rows = db.execute(text(sql), params).mappings().all()
    mids = [r["mid"] for r in rows]
    sources: dict[str, list[PostSource]] = {}
    if mids:
        for src in db.query(PostSource).filter(PostSource.mid.in_(mids)).all():
            sources.setdefault(src.mid, []).append(src)

    items = [
        PostListItem(
            mid=r["mid"],
            text=r["text"],
            author_uid=r["author_uid"],
            author_nickname=r["author_nickname"] or "",
            created_at=r["created_at"],
            reposts=r["reposts"],
            comments=r["comments"],
            likes=r["likes"],
            interactions=r["reposts"] + r["comments"] + r["likes"],
            is_repost=bool(r["is_repost"]),
            sentiment=r["sentiment"],
            sentiment_source=r["sentiment_source"] or "unscored",
            sources=[SourceOut.model_validate(s) for s in sources.get(r["mid"], [])],
        )
        for r in rows
    ]
    return PostPage(items=items, total=total, page=page, page_size=page_size)


@router.get("/{mid}", response_model=PostDetail)
def post_detail(mid: str, db: Session = Depends(get_db)):
    post = db.query(Post).filter(Post.mid == mid).first()
    if not post:
        raise HTTPException(status_code=404, detail="帖子不存在")
    author = db.query(Author).filter(Author.uid == post.author_uid).first()
    label, source = get_final(db, "post", mid)
    sources = db.query(PostSource).filter(PostSource.mid == mid).all()

    comments = db.query(Comment).filter(Comment.mid == mid).order_by(Comment.created_at.asc()).all()
    comment_labels = final_map(db, "comment", [c.id for c in comments])
    comment_authors = {
        a.uid: a.nickname
        for a in db.query(Author).filter(Author.uid.in_([c.author_uid for c in comments if c.author_uid])).all()
    } if comments else {}
    detail_comments = [
        CommentOut(
            id=c.id,
            mid=c.mid,
            text=c.text,
            author_uid=c.author_uid,
            author_nickname=comment_authors.get(c.author_uid or "", ""),
            created_at=c.created_at,
            likes=c.likes,
            sentiment=comment_labels.get(c.id, (None, "unscored"))[0],
            sentiment_source=comment_labels.get(c.id, (None, "unscored"))[1],
        )
        for c in comments
    ]

    reposts = db.query(Repost).filter(Repost.root_mid == mid).order_by(Repost.created_at.asc()).all()
    repost_labels = final_map(db, "repost", [r.mid for r in reposts])
    repost_authors = {
        a.uid: a.nickname
        for a in db.query(Author).filter(Author.uid.in_([r.author_uid for r in reposts if r.author_uid])).all()
    } if reposts else {}
    detail_reposts = [
        RepostOut(
            mid=r.mid,
            root_mid=r.root_mid,
            text=r.text,
            author_uid=r.author_uid,
            author_nickname=repost_authors.get(r.author_uid or "", ""),
            created_at=r.created_at,
            sentiment=repost_labels.get(r.mid, (None, "unscored"))[0],
            sentiment_source=repost_labels.get(r.mid, (None, "unscored"))[1],
        )
        for r in reposts
    ]

    metrics = (
        db.query(PostMetric)
        .filter(PostMetric.mid == mid)
        .order_by(PostMetric.ts.asc())
        .all()
    )
    return PostDetail(
        mid=post.mid,
        text=post.text,
        author_uid=post.author_uid,
        author_nickname=author.nickname if author else "",
        created_at=post.created_at,
        reposts=post.reposts,
        comments=post.comments,
        likes=post.likes,
        interactions=post.reposts + post.comments + post.likes,
        is_repost=post.is_repost,
        sentiment=label,
        sentiment_source=source,
        sources=[SourceOut.model_validate(s) for s in sources],
        raw=post.raw,
        fetched_at=post.fetched_at,
        detail_comments=detail_comments,
        detail_reposts=detail_reposts,
        metrics=[
            MetricPoint(ts=m.ts, reposts=m.reposts, comments=m.comments, likes=m.likes)
            for m in metrics
        ],
    )
