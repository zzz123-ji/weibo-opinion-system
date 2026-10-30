"""微博客户端：Mock/HTTP 两种实现与账号轮换。"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import httpx
from sqlalchemy.orm import Session

from app.config import settings
from app.models import WeiboCookie
from app.security import decrypt_cookie

from . import parsers

logger = logging.getLogger(__name__)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class CrawlError(RuntimeError):
    pass


class RateLimited(CrawlError):
    pass


class LoginRequired(CrawlError):
    pass


class NoAvailableAccount(CrawlError):
    pass


class MockWeiboClient:
    """读取 fixtures 的离线客户端，用于本地演示和解析单测。"""

    def __init__(self, fixtures: Path | None = None):
        self.fixtures = fixtures or FIXTURES

    def _load(self, name: str) -> dict:
        return json.loads((self.fixtures / name).read_text(encoding="utf-8"))

    def search_keyword(self, keyword: str, page: int = 1) -> list[dict]:
        if page > 1:
            return []
        return parsers.parse_search(self._load("search.json"), keyword)

    def comments(self, mid: str, max_id: str | None = None) -> list[dict]:
        if max_id:
            return []
        return parsers.parse_comments(self._load("comments.json"), mid)

    def reposts(self, mid: str) -> list[dict]:
        return parsers.parse_reposts(self._load("reposts.json"), mid)

    def user_timeline(self, uid: str, page: int = 1) -> list[dict]:
        if page > 1:
            return []
        return parsers.parse_user_timeline(self._load("timeline.json"), uid)

    def hot_search(self) -> list[dict]:
        return parsers.parse_hotsearch(self._load("hotsearch.json"))


class HttpWeiboClient:
    """m.weibo.cn 移动端 JSON 客户端。

    Cookie 由管理员通过设置页导入；本类不实现登录、验证码或反爬绕过。
    """

    MOBILE_UA = (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
    )

    def __init__(
        self,
        cookie: str,
        min_interval: float | None = None,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cookie = cookie
        self.min_interval = settings.crawl_rate_limit_seconds if min_interval is None else min_interval
        self._last_request = 0.0
        self._sleep = sleeper
        self._clock = clock
        self.client = client or httpx.Client(
            timeout=20,
            follow_redirects=True,
            headers={
                "User-Agent": self.MOBILE_UA,
                "Referer": "https://m.weibo.cn/",
                "Accept": "application/json, text/plain, */*",
            },
        )

    def close(self) -> None:
        self.client.close()

    def _wait(self) -> None:
        delta = self._clock() - self._last_request
        if delta < self.min_interval:
            self._sleep(self.min_interval - delta)
        self._last_request = self._clock()

    def request_json(self, url: str, params: dict | None = None) -> dict:
        self._wait()
        headers = {"Cookie": self.cookie}
        resp = self.client.get(url, params=params or {}, headers=headers)
        if resp.status_code in (418, 403):
            raise RateLimited(f"HTTP {resp.status_code}，账号需要冷却")
        if resp.status_code in (401, 302):
            raise LoginRequired(f"HTTP {resp.status_code}，Cookie 可能失效")
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError as exc:
            raise CrawlError(f"返回不是 JSON：{resp.text[:120]}") from exc
        if payload.get("ok") == 0:
            message = str(payload.get("msg") or payload.get("message") or "")
            url_hint = str(payload.get("url") or "")
            if "登录" in message or "login" in url_hint.lower():
                raise LoginRequired(message or "Cookie 可能失效")
            raise CrawlError(message or "接口返回 ok=0")
        return payload

    def search_keyword(self, keyword: str, page: int = 1) -> list[dict]:
        payload = self.request_json(
            "https://m.weibo.cn/api/container/getIndex",
            {"containerid": f"100103type=1&q={keyword}", "page": page},
        )
        return parsers.parse_search(payload, keyword)

    def comments(self, mid: str, max_id: str | None = None) -> list[dict]:
        params = {"id": mid, "mid": mid}
        if max_id:
            params["max_id"] = max_id
        payload = self.request_json("https://m.weibo.cn/comments/hotflow", params)
        return parsers.parse_comments(payload, mid)

    def reposts(self, mid: str) -> list[dict]:
        payload = self.request_json(
            "https://m.weibo.cn/api/statuses/repostTimeline",
            {"id": mid, "page": 1},
        )
        return parsers.parse_reposts(payload, mid)

    def user_timeline(self, uid: str, page: int = 1) -> list[dict]:
        payload = self.request_json(
            "https://m.weibo.cn/api/container/getIndex",
            {"containerid": f"107603{uid}", "page": page},
        )
        return parsers.parse_user_timeline(payload, uid)

    def hot_search(self) -> list[dict]:
        payload = self.request_json("https://weibo.com/ajax/side/hotSearch")
        return parsers.parse_hotsearch(payload)


class AccountPool:
    """账号池：最小请求间隔、418/403 冷却、连续失败标记异常。"""

    def __init__(
        self,
        db: Session,
        min_interval: float | None = None,
        cooldown_minutes: int = 30,
        max_failures: int = 3,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.db = db
        self.min_interval = settings.crawl_rate_limit_seconds if min_interval is None else min_interval
        self.cooldown_minutes = cooldown_minutes
        self.max_failures = max_failures
        self._sleep = sleeper
        self._clock = clock

    def acquire(self) -> WeiboCookie:
        now = datetime.utcnow()
        rows = (
            self.db.query(WeiboCookie)
            .filter(WeiboCookie.status.in_(("ok", "cooling")))
            .order_by(WeiboCookie.last_used_at.asc().nullsfirst(), WeiboCookie.id.asc())
            .all()
        )
        for row in rows:
            if row.status == "cooling" and row.cooling_until and row.cooling_until <= now:
                row.status = "ok"
                row.cooling_until = None
                row.fail_count = 0
                self.db.commit()
            if row.cooling_until and row.cooling_until > now:
                continue
            last = row.last_used_at
            if last:
                delta = (now - last).total_seconds()
                if delta < self.min_interval:
                    self._sleep(self.min_interval - delta)
            row.last_used_at = datetime.utcnow()
            self.db.commit()
            return row
        raise NoAvailableAccount("没有可用微博账号；请检查 Cookie 状态或冷却时间")

    def report_rate_limited(self, cookie: WeiboCookie, code: str = "418") -> None:
        cookie.cooling_until = datetime.utcnow() + timedelta(minutes=self.cooldown_minutes)
        cookie.fail_count += 1
        cookie.status = "abnormal" if cookie.fail_count >= self.max_failures else "cooling"
        self.db.commit()

    def report_login_required(self, cookie: WeiboCookie) -> None:
        cookie.status = "need_login"
        self.db.commit()

    def report_success(self, cookie: WeiboCookie) -> None:
        if cookie.status == "ok":
            cookie.fail_count = 0
            self.db.commit()


class RotatingWeiboClient:
    """每次调用从账号池取账号，异常时冷却/标记，再换下一个账号。"""

    def __init__(self, db: Session, pool: AccountPool | None = None):
        self.db = db
        self.pool = pool or AccountPool(db)

    def _call(self, method: str, *args, **kwargs):
        last_error: Exception | None = None
        for _attempt in range(3):
            cookie = self.pool.acquire()
            client = HttpWeiboClient(decrypt_cookie(cookie.cookie_enc), min_interval=self.pool.min_interval)
            try:
                result = getattr(client, method)(*args, **kwargs)
                self.pool.report_success(cookie)
                return result
            except RateLimited as exc:
                self.pool.report_rate_limited(cookie)
                last_error = exc
                logger.warning("账号 %s 被限流，切换下一个账号", cookie.account_name)
            except LoginRequired as exc:
                self.pool.report_login_required(cookie)
                last_error = exc
                logger.warning("账号 %s Cookie 失效，切换下一个账号", cookie.account_name)
            finally:
                client.close()
        raise last_error or CrawlError("采集失败")

    def search_keyword(self, keyword: str, page: int = 1):
        return self._call("search_keyword", keyword, page)

    def comments(self, mid: str, max_id: str | None = None):
        return self._call("comments", mid, max_id)

    def reposts(self, mid: str):
        return self._call("reposts", mid)

    def user_timeline(self, uid: str, page: int = 1):
        return self._call("user_timeline", uid, page)

    def hot_search(self):
        return self._call("hot_search")
