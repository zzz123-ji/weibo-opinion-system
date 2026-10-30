"""采集日志。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import require_admin
from ..models import AppUser, CrawlLog
from ..schemas import CrawlLogOut

router = APIRouter(prefix="/crawl", tags=["crawl"])


@router.get("/logs", response_model=list[CrawlLogOut])
def logs(
    task_type: str | None = None,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    _admin: AppUser = Depends(require_admin),
):
    q = db.query(CrawlLog)
    if task_type:
        q = q.filter(CrawlLog.task_type == task_type)
    if status:
        q = q.filter(CrawlLog.status == status)
    return q.order_by(CrawlLog.started_at.desc()).limit(limit).all()
