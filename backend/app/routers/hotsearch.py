"""热搜快照与关键词命中。"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..models import HotSearchSnapshot
from ..schemas import HotSearchOut

router = APIRouter(prefix="/hotsearch", tags=["hotsearch"], dependencies=[Depends(get_current_user)])


@router.get("", response_model=list[HotSearchOut])
def list_hotsearch(
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    only_matched: bool = False,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    q = db.query(HotSearchSnapshot)
    if time_from:
        q = q.filter(HotSearchSnapshot.ts >= time_from)
    if time_to:
        q = q.filter(HotSearchSnapshot.ts <= time_to)
    if only_matched:
        q = q.filter(HotSearchSnapshot.matched_keyword_id.isnot(None))
    return q.order_by(HotSearchSnapshot.ts.desc(), HotSearchSnapshot.rank.asc()).limit(limit).all()
