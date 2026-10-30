"""引擎评估。"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user
from ..schemas import EvalOut
from ..services.metrics import evaluate_engines

router = APIRouter(prefix="/eval", tags=["eval"], dependencies=[Depends(get_current_user)])


@router.get("", response_model=EvalOut)
def eval_engines(db: Session = Depends(get_db)):
    return EvalOut(**evaluate_engines(db))
