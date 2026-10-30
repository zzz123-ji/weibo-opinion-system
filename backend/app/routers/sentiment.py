"""情感队列：历史重跑与手动消费。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import require_admin
from ..models import AppUser
from ..schemas import Message, RerunIn
from ..services.queue import enqueue_all, queue_counts
from ..services.sentiment import available_engines
from ..services.worker import process_queue

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


@router.post("/rerun", response_model=Message)
def rerun(
    payload: RerunIn,
    db: Session = Depends(get_db),
    _admin: AppUser = Depends(require_admin),
):
    if payload.engine not in available_engines():
        raise HTTPException(status_code=400, detail=f"引擎不可用：{payload.engine}")
    n = enqueue_all(db, payload.text_type, payload.time_from, payload.time_to, payload.limit)
    return Message(message=f"已入队 {n} 条；sentiment worker 会按当前引擎处理")


@router.post("/run-once", response_model=Message)
def run_once(
    batch_size: int = 64,
    max_batches: int = 10,
    db: Session = Depends(get_db),
    _admin: AppUser = Depends(require_admin),
):
    n = process_queue(db, batch_size=batch_size, max_batches=max_batches)
    return Message(message=f"本次处理 {n} 条")


@router.get("/queue")
def queue_status(
    db: Session = Depends(get_db),
    _admin: AppUser = Depends(require_admin),
):
    return queue_counts(db)
