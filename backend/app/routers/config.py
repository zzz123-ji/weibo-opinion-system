"""配置管理：关键词、监控账号、Cookie、用户、引擎、重跑、采集日志。"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import get_current_user, require_admin
from ..models import (
    AppUser,
    CrawlLog,
    Keyword,
    Setting,
    WatchAccount,
    WeiboCookie,
)
from ..schemas import (
    AccountCreate,
    AccountOut,
    AccountUpdate,
    CookieManualCreate,
    CookieOut,
    CrawlLogOut,
    EngineConfigIn,
    EngineConfigOut,
    KeywordCreate,
    KeywordOut,
    KeywordUpdate,
    Message,
    RerunIn,
    UserCreate,
    UserOut,
    UserUpdate,
)
from ..security import encrypt_cookie, hash_password
from ..services.queue import enqueue_all
from ..services.sentiment import available_engines

router = APIRouter(prefix="/config", tags=["config"], dependencies=[Depends(get_current_user)])


def _setting_str(db: Session, key: str, default: str = "") -> str:
    row = db.get(Setting, key)
    if not row or row.value is None:
        return default
    value = row.value
    if isinstance(value, str):
        return value.strip('"')
    return str(value)


@router.get("/keywords", response_model=list[KeywordOut])
def list_keywords(
    enabled: bool | None = None,
    db: Session = Depends(get_db),
    _user: AppUser = Depends(get_current_user),
):
    q = db.query(Keyword)
    if enabled is not None:
        q = q.filter(Keyword.enabled.is_(enabled))
    return q.order_by(Keyword.id.asc()).all()


@router.post("/keywords", response_model=KeywordOut)
def create_keyword(
    payload: KeywordCreate,
    db: Session = Depends(get_db),
    user: AppUser = Depends(require_admin),
):
    if db.query(Keyword).filter(Keyword.word == payload.word).first():
        raise HTTPException(status_code=409, detail="关键词已存在")
    row = Keyword(**payload.model_dump(), created_by=user.id)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.put("/keywords/{keyword_id}", response_model=KeywordOut)
def update_keyword(
    keyword_id: int,
    payload: KeywordUpdate,
    db: Session = Depends(get_db),
    _user: AppUser = Depends(require_admin),
):
    row = db.get(Keyword, keyword_id)
    if not row:
        raise HTTPException(status_code=404, detail="关键词不存在")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(row, key, value)
    db.commit()
    db.refresh(row)
    return row


@router.delete("/keywords/{keyword_id}", response_model=Message)
def delete_keyword(
    keyword_id: int,
    db: Session = Depends(get_db),
    _user: AppUser = Depends(require_admin),
):
    row = db.get(Keyword, keyword_id)
    if not row:
        raise HTTPException(status_code=404, detail="关键词不存在")
    # 热搜快照可能引用该关键词，先解除引用再删除。
    from ..models import HotSearchSnapshot

    db.query(HotSearchSnapshot).filter(
        HotSearchSnapshot.matched_keyword_id == keyword_id
    ).update({HotSearchSnapshot.matched_keyword_id: None}, synchronize_session=False)
    db.delete(row)
    db.commit()
    return Message(message="已删除")


@router.get("/accounts", response_model=list[AccountOut])
def list_accounts(
    db: Session = Depends(get_db),
    _user: AppUser = Depends(get_current_user),
):
    return db.query(WatchAccount).order_by(WatchAccount.id.asc()).all()


@router.post("/accounts", response_model=AccountOut)
def create_account(
    payload: AccountCreate,
    db: Session = Depends(get_db),
    _user: AppUser = Depends(require_admin),
):
    if db.query(WatchAccount).filter(WatchAccount.uid == payload.uid).first():
        raise HTTPException(status_code=409, detail="账号已存在")
    row = WatchAccount(**payload.model_dump())
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.put("/accounts/{account_id}", response_model=AccountOut)
def update_account(
    account_id: int,
    payload: AccountUpdate,
    db: Session = Depends(get_db),
    _user: AppUser = Depends(require_admin),
):
    row = db.get(WatchAccount, account_id)
    if not row:
        raise HTTPException(status_code=404, detail="账号不存在")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(row, key, value)
    db.commit()
    db.refresh(row)
    return row


@router.delete("/accounts/{account_id}", response_model=Message)
def delete_account(
    account_id: int,
    db: Session = Depends(get_db),
    _user: AppUser = Depends(require_admin),
):
    row = db.get(WatchAccount, account_id)
    if not row:
        raise HTTPException(status_code=404, detail="账号不存在")
    db.delete(row)
    db.commit()
    return Message(message="已删除")


@router.get("/users", response_model=list[UserOut])
def list_users(
    db: Session = Depends(get_db),
    _admin: AppUser = Depends(require_admin),
):
    return db.query(AppUser).order_by(AppUser.id.asc()).all()


@router.post("/users", response_model=UserOut)
def create_user(
    payload: UserCreate,
    db: Session = Depends(get_db),
    _admin: AppUser = Depends(require_admin),
):
    if db.query(AppUser).filter(AppUser.username == payload.username).first():
        raise HTTPException(status_code=409, detail="用户名已存在")
    row = AppUser(
        username=payload.username,
        password_hash=hash_password(payload.password),
        role=payload.role,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.put("/users/{user_id}", response_model=UserOut)
def update_user(
    user_id: int,
    payload: UserUpdate,
    db: Session = Depends(get_db),
    admin: AppUser = Depends(require_admin),
):
    row = db.get(AppUser, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="用户不存在")
    if payload.password:
        row.password_hash = hash_password(payload.password)
    if payload.role:
        row.role = payload.role
    if payload.enabled is not None:
        if row.id == admin.id and not payload.enabled:
            raise HTTPException(status_code=400, detail="不能禁用自己")
        row.enabled = payload.enabled
    db.commit()
    db.refresh(row)
    return row


@router.delete("/users/{user_id}", response_model=Message)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: AppUser = Depends(require_admin),
):
    row = db.get(AppUser, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="用户不存在")
    if row.id == admin.id:
        raise HTTPException(status_code=400, detail="不能删除自己")
    from ..models import SentimentLabel

    if db.query(SentimentLabel).filter(SentimentLabel.user_id == row.id).count() > 0:
        raise HTTPException(status_code=400, detail="该用户已有标注记录，请禁用而不是删除")
    db.delete(row)
    db.commit()
    return Message(message="已删除")


@router.get("/cookies", response_model=list[CookieOut])
def list_cookies(
    db: Session = Depends(get_db),
    _admin: AppUser = Depends(require_admin),
):
    return db.query(WeiboCookie).order_by(WeiboCookie.id.asc()).all()


@router.post("/cookies", response_model=CookieOut)
def create_cookie(
    payload: CookieManualCreate,
    db: Session = Depends(get_db),
    _admin: AppUser = Depends(require_admin),
):
    row = db.query(WeiboCookie).filter(WeiboCookie.account_name == payload.account_name).first()
    if row:
        row.cookie_enc = encrypt_cookie(payload.cookie)
        row.status = "ok"
        row.cooling_until = None
        row.fail_count = 0
    else:
        row = WeiboCookie(
            account_name=payload.account_name,
            cookie_enc=encrypt_cookie(payload.cookie),
            status="ok",
        )
        db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.post("/cookies/qrcode", response_model=Message)
def qrcode_login(_admin: AppUser = Depends(require_admin)):
    """扫码登录入口。

    真实扫码需要 Playwright + 浏览器 + 人工扫码，不能在无头测试环境伪造。
    未安装 Playwright 时明确返回 501，并提示使用手动 Cookie 导入。
    """
    try:
        import playwright  # noqa: F401
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="未安装 Playwright。请先 pip install playwright && playwright install chromium；"
                   "或使用 POST /config/cookies 手动导入 Cookie。",
        )
    raise HTTPException(
        status_code=501,
        detail="扫码登录需要人工在浏览器中完成，当前部署未启用交互式浏览器；"
               "请使用 POST /config/cookies 手动导入 Cookie。",
    )


@router.delete("/cookies/{cookie_id}", response_model=Message)
def delete_cookie(
    cookie_id: int,
    db: Session = Depends(get_db),
    _admin: AppUser = Depends(require_admin),
):
    row = db.get(WeiboCookie, cookie_id)
    if not row:
        raise HTTPException(status_code=404, detail="Cookie 不存在")
    db.delete(row)
    db.commit()
    return Message(message="已删除")


@router.get("/engine", response_model=EngineConfigOut)
def get_engine(
    db: Session = Depends(get_db),
    _user: AppUser = Depends(get_current_user),
):
    return EngineConfigOut(
        active_engine=_setting_str(db, "active_engine", "local"),
        active_version=_setting_str(db, "active_engine_version", ""),
        available=available_engines(),
    )


@router.post("/engine", response_model=EngineConfigOut)
def set_engine(
    payload: EngineConfigIn,
    db: Session = Depends(get_db),
    _admin: AppUser = Depends(require_admin),
):
    if payload.name not in available_engines():
        raise HTTPException(status_code=400, detail=f"引擎不可用：{payload.name}")
    db.merge(Setting(key="active_engine", value=payload.name))
    db.merge(Setting(key="active_engine_version", value=payload.version or ""))
    db.commit()
    return EngineConfigOut(
        active_engine=payload.name,
        active_version=payload.version or "",
        available=available_engines(),
    )


@router.post("/sentiment/rerun", response_model=Message)
def rerun_sentiment(
    payload: RerunIn,
    db: Session = Depends(get_db),
    _admin: AppUser = Depends(require_admin),
):
    if payload.engine not in available_engines():
        raise HTTPException(status_code=400, detail=f"引擎不可用：{payload.engine}")
    n = enqueue_all(db, payload.text_type, payload.time_from, payload.time_to, payload.limit)
    return Message(message=f"已入队 {n} 条；请由 sentiment worker 处理")


@router.get("/crawl/logs", response_model=list[CrawlLogOut])
def crawl_logs(
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
