"""采集任务：关键词、账号、评论、转发、热搜。"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.config import settings
from app.models import (
    Author,
    Comment,
    CrawlLog,
    HotSearchSnapshot,
    Keyword,
    Post,
    PostMetric,
    PostSource,
    Repost,
    WatchAccount,
)
from app.services.queue import enqueue_text

from .client import CrawlError, MockWeiboClient

logger = logging.getLogger(__name__)


class CrawlService:
    def __init__(self, db: Session, client, max_pages: int | None = None, repost_threshold: int | None = None):
        self.db = db
        self.client = client
        self.max_pages = max_pages or settings.crawl_max_pages
        self.repost_threshold = (
            settings.crawl_repost_threshold if repost_threshold is None else repost_threshold
        )

    # ---------- 基础写入 ----------
    def _start_log(self, task_type: str, target: str) -> CrawlLog:
        log = CrawlLog(task_type=task_type, target=target, status="running")
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        return log

    def _finish_log(self, log: CrawlLog, status: str, items: int, error: str | None = None) -> None:
        log.status = status
        log.items = items
        log.finished_at = datetime.utcnow()
        if error:
            log.error_msg = error[:1000]
        self.db.commit()

    def _upsert_author(self, data: dict) -> None:
        uid = data.get("uid") or ""
        if not uid:
            return
        author = self.db.get(Author, uid)
        if author:
            author.nickname = data.get("nickname") or author.nickname
            author.followers = max(author.followers, int(data.get("followers") or 0))
            author.verified = bool(data.get("verified")) or author.verified
            author.updated_at = datetime.utcnow()
        else:
            self.db.add(
                Author(
                    uid=uid,
                    nickname=data.get("nickname") or "",
                    followers=int(data.get("followers") or 0),
                    verified=bool(data.get("verified")),
                )
            )

    def _upsert_post(self, item: dict) -> bool:
        """返回 True 表示新帖；False 表示已存在（用于增量停止）。"""
        self._upsert_author(item.get("author") or {})
        mid = item["mid"]
        existing = self.db.get(Post, mid)
        if existing:
            existing.reposts = max(existing.reposts, int(item.get("reposts") or 0))
            existing.comments = max(existing.comments, int(item.get("comments") or 0))
            existing.likes = max(existing.likes, int(item.get("likes") or 0))
            existing.fetched_at = datetime.utcnow()
            return False
        post = Post(
            mid=mid,
            text=item.get("text") or "",
            author_uid=(item.get("author") or {}).get("uid") or None,
            created_at=item.get("created_at") or datetime.utcnow(),
            reposts=int(item.get("reposts") or 0),
            comments=int(item.get("comments") or 0),
            likes=int(item.get("likes") or 0),
            is_repost=bool(item.get("is_repost")),
            root_mid=item.get("root_mid"),
            raw=item.get("raw"),
            fetched_at=datetime.utcnow(),
        )
        self.db.add(post)
        self.db.add(
            PostMetric(
                mid=mid,
                ts=post.created_at,
                reposts=post.reposts,
                comments=post.comments,
                likes=post.likes,
            )
        )
        source_type = item.get("source_type") or ""
        source_value = item.get("source_value") or ""
        if source_type:
            exists = (
                self.db.query(PostSource)
                .filter(
                    PostSource.mid == mid,
                    PostSource.source_type == source_type,
                    PostSource.source_value == source_value,
                )
                .first()
            )
            if not exists:
                self.db.add(
                    PostSource(
                        mid=mid,
                        source_type=source_type,
                        source_value=source_value,
                        first_seen_at=datetime.utcnow(),
                    )
                )
        enqueue_text(self.db, "post", mid)
        return True

    def _upsert_comment(self, item: dict) -> bool:
        self._upsert_author(item.get("author") or {})
        cid = item["id"]
        if self.db.get(Comment, cid):
            return False
        self.db.add(
            Comment(
                id=cid,
                mid=item["mid"],
                text=item.get("text") or "",
                author_uid=(item.get("author") or {}).get("uid") or None,
                created_at=item.get("created_at") or datetime.utcnow(),
                likes=int(item.get("likes") or 0),
                reply_to=item.get("reply_to"),
                raw=item.get("raw"),
            )
        )
        enqueue_text(self.db, "comment", cid)
        return True

    def _upsert_repost(self, item: dict) -> bool:
        self._upsert_author(item.get("author") or {})
        mid = item["mid"]
        if self.db.get(Repost, mid):
            return False
        self.db.add(
            Repost(
                mid=mid,
                root_mid=item["root_mid"],
                text=item.get("text") or "",
                author_uid=(item.get("author") or {}).get("uid") or None,
                created_at=item.get("created_at") or datetime.utcnow(),
                raw=item.get("raw"),
            )
        )
        enqueue_text(self.db, "repost", mid)
        return True

    # ---------- 任务 ----------
    def run_keyword(self, word: str) -> int:
        log = self._start_log("keyword", word)
        count = 0
        try:
            for page in range(1, self.max_pages + 1):
                items = self.client.search_keyword(word, page)
                if not items:
                    break
                stop = False
                for item in items:
                    if self.db.get(Post, item["mid"]):
                        stop = True
                        break
                    if self._upsert_post(item):
                        count += 1
                self.db.commit()
                if stop:
                    break
            self._finish_log(log, "success", count)
            return count
        except CrawlError as exc:
            self._finish_log(log, "failed", count, str(exc))
            raise

    def run_account(self, uid: str) -> int:
        log = self._start_log("account", uid)
        count = 0
        try:
            for page in range(1, self.max_pages + 1):
                items = self.client.user_timeline(uid, page)
                if not items:
                    break
                stop = False
                for item in items:
                    if self.db.get(Post, item["mid"]):
                        stop = True
                        break
                    if self._upsert_post(item):
                        count += 1
                self.db.commit()
                if stop:
                    break
            self._finish_log(log, "success", count)
            return count
        except CrawlError as exc:
            self._finish_log(log, "failed", count, str(exc))
            raise

    def run_hotsearch(self) -> int:
        log = self._start_log("hotsearch", "weibo.com/ajax/side/hotSearch")
        count = 0
        try:
            items = self.client.hot_search()
            ts = datetime.utcnow().replace(second=0, microsecond=0)
            active_keywords = self.db.query(Keyword).filter(Keyword.enabled.is_(True)).all()
            for item in items:
                matched = next((k for k in active_keywords if k.word and k.word in item["word"]), None)
                existing = (
                    self.db.query(HotSearchSnapshot)
                    .filter(HotSearchSnapshot.ts == ts, HotSearchSnapshot.rank == item["rank"])
                    .first()
                )
                if existing:
                    existing.word = item["word"]
                    existing.heat = item["heat"]
                    existing.label = item["label"]
                    existing.matched_keyword_id = matched.id if matched else None
                else:
                    self.db.add(
                        HotSearchSnapshot(
                            ts=ts,
                            rank=item["rank"],
                            word=item["word"],
                            heat=item["heat"],
                            label=item["label"],
                            matched_keyword_id=matched.id if matched else None,
                        )
                    )
                    count += 1
                if matched:
                    # 命中监控关键词的热搜词，临时加入关键词搜索 24 小时。
                    temp_word = item["word"]
                    row = self.db.query(Keyword).filter(Keyword.word == temp_word).first()
                    if not row:
                        self.db.add(
                            Keyword(
                                word=temp_word,
                                enabled=True,
                                is_temp=True,
                                expire_at=datetime.utcnow() + timedelta(hours=24),
                            )
                        )
            self.db.commit()
            self._finish_log(log, "success", count)
            return count
        except CrawlError as exc:
            self._finish_log(log, "failed", count, str(exc))
            raise

    def run_comments(self, mid: str) -> int:
        log = self._start_log("comments", mid)
        count = 0
        try:
            items = self.client.comments(mid)
            for item in items:
                if self._upsert_comment(item):
                    count += 1
            self.db.commit()
            self._finish_log(log, "success", count)
            return count
        except CrawlError as exc:
            self._finish_log(log, "failed", count, str(exc))
            raise

    def run_reposts(self, mid: str) -> int:
        post = self.db.get(Post, mid)
        if not post or post.reposts < self.repost_threshold:
            return 0
        log = self._start_log("reposts", mid)
        count = 0
        try:
            items = self.client.reposts(mid)
            for item in items:
                if self._upsert_repost(item):
                    count += 1
            self.db.commit()
            self._finish_log(log, "success", count)
            return count
        except CrawlError as exc:
            self._finish_log(log, "failed", count, str(exc))
            raise

    def run_all_once(self) -> dict[str, int]:
        result = {"keyword": 0, "account": 0, "hotsearch": 0, "comments": 0, "reposts": 0}
        now = datetime.utcnow()
        for kw in self.db.query(Keyword).filter(Keyword.enabled.is_(True)).all():
            if kw.expire_at and kw.expire_at < now:
                continue
            try:
                result["keyword"] += self.run_keyword(kw.word)
            except CrawlError as exc:
                logger.warning("关键词 %s 采集失败：%s", kw.word, exc)
        for acc in self.db.query(WatchAccount).filter(WatchAccount.enabled.is_(True)).all():
            try:
                result["account"] += self.run_account(acc.uid)
            except CrawlError as exc:
                logger.warning("账号 %s 采集失败：%s", acc.uid, exc)
        try:
            result["hotsearch"] += self.run_hotsearch()
        except CrawlError as exc:
            logger.warning("热搜采集失败：%s", exc)
        # 只对最近帖子补互动，避免每轮全量抓取。
        since = now - timedelta(hours=24)
        recent = (
            self.db.query(Post)
            .filter(Post.created_at >= since)
            .order_by(Post.reposts.desc())
            .limit(30)
            .all()
        )
        for post in recent:
            try:
                result["comments"] += self.run_comments(post.mid)
                result["reposts"] += self.run_reposts(post.mid)
            except CrawlError as exc:
                logger.warning("互动采集失败 %s：%s", post.mid, exc)
        return result
