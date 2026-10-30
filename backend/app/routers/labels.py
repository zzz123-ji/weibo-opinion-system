"""标注工作台接口。"""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import AppUser, SentimentLabel
from ..schemas import (
    AutoLabelOut,
    LabelNextOut,
    LabelPutRequest,
    LabelPutResponse,
    LabelStatsOut,
)
from ..services.final import get_final, get_text
from ..services.sampling import label_stats, next_sample

router = APIRouter(prefix="/label", tags=["label"])


@router.get("/next", response_model=LabelNextOut)
def next_label(
    strategy: str = Query(default="random", pattern="^(random|low_confidence|disagreement)$"),
    text_type: str | None = Query(default=None, pattern="^(post|comment|repost)$"),
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    keyword: str | None = None,
    db: Session = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    sample = next_sample(db, user.id, strategy, text_type, time_from, time_to, keyword)
    if not sample:
        raise HTTPException(status_code=404, detail="没有待标注样本")
    final_label, final_source = get_final(db, sample["text_type"], sample["text_id"])
    stats = label_stats(db, user.id)
    return LabelNextOut(
        text_type=sample["text_type"],
        text_id=sample["text_id"],
        text=sample["text"],
        post_context=sample["post_context"],
        auto_labels=[
            AutoLabelOut(
                engine=a.engine,
                engine_version=a.engine_version,
                label=a.label,
                score=a.score,
                created_at=a.created_at,
            )
            for a in sample["auto_labels"]
        ],
        final_label=final_label,
        final_source=final_source,
        my_label=None,
        today_count=stats["my_today"],
        total_count=stats["my_total"],
    )


@router.put("/{text_type}/{text_id}", response_model=LabelPutResponse)
def put_label(
    text_type: str,
    text_id: str,
    payload: LabelPutRequest,
    db: Session = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    if text_type not in {"post", "comment", "repost"}:
        raise HTTPException(status_code=400, detail="text_type 必须是 post/comment/repost")
    if get_text(db, text_type, text_id) is None:
        raise HTTPException(status_code=404, detail="文本不存在")
    row = (
        db.query(SentimentLabel)
        .filter(
            SentimentLabel.text_type == text_type,
            SentimentLabel.text_id == text_id,
            SentimentLabel.user_id == user.id,
        )
        .first()
    )
    if row:
        row.label = payload.label
        row.updated_at = datetime.utcnow()
    else:
        row = SentimentLabel(
            text_type=text_type,
            text_id=text_id,
            user_id=user.id,
            label=payload.label,
        )
        db.add(row)
    db.commit()
    final_label, final_source = get_final(db, text_type, text_id)
    return LabelPutResponse(
        text_type=text_type,
        text_id=text_id,
        label=payload.label,
        final_label=final_label,
        final_source=final_source,
    )


@router.get("/stats", response_model=LabelStatsOut)
def stats(
    db: Session = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    return LabelStatsOut(**label_stats(db, user.id))


@router.get("/export")
def export_labels(
    db: Session = Depends(get_db),
    user: AppUser = Depends(get_current_user),
):
    rows = (
        db.query(SentimentLabel)
        .order_by(SentimentLabel.text_type, SentimentLabel.text_id, SentimentLabel.updated_at)
        .all()
    )

    def generate():
        for row in rows:
            text_info = get_text(db, row.text_type, row.text_id)
            if not text_info:
                continue
            text, context = text_info
            yield json.dumps(
                {
                    "text": text,
                    "label": row.label,
                    "text_type": row.text_type,
                    "text_id": row.text_id,
                    "user": row.user_id,
                    "context": context,
                },
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "attachment; filename=labels.jsonl"},
    )
