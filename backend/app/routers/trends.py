"""趋势、情感占比、负面突增、Top 负面帖。"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..schemas import AlertOut, SentimentSeries, TopNegativeOut, VolumeSeries
from ..services import trends as trends_service

router = APIRouter(prefix="/trend", tags=["trend"], dependencies=[Depends(get_current_user)])


@router.get("/volume", response_model=VolumeSeries)
def volume(
    granularity: str = Query(default="day", pattern="^(hour|day)$"),
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    keywords: list[str] | None = Query(default=None),
    split_by_keyword: bool = False,
    db: Session = Depends(get_db),
):
    return trends_service.volume_series(
        db, granularity, time_from, time_to, keywords, split_by_keyword
    )


@router.get("/sentiment", response_model=SentimentSeries)
def sentiment(
    granularity: str = Query(default="day", pattern="^(hour|day)$"),
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    keywords: list[str] | None = Query(default=None),
    db: Session = Depends(get_db),
):
    return trends_service.sentiment_series(db, granularity, time_from, time_to, keywords)


@router.get("/alerts", response_model=list[AlertOut])
def alerts(
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    threshold_ratio: float = Query(default=3.0, gt=1.0),
    min_count: int = Query(default=10, ge=1),
    db: Session = Depends(get_db),
):
    return trends_service.negative_alerts(
        db, time_from, time_to, threshold_ratio, min_count
    )


@router.get("/top-negative", response_model=list[TopNegativeOut])
def top_negative(
    limit: int = Query(default=10, ge=1, le=50),
    time_from: datetime | None = None,
    time_to: datetime | None = None,
    keywords: list[str] | None = Query(default=None),
    db: Session = Depends(get_db),
):
    return trends_service.top_negative(db, limit, time_from, time_to, keywords)
