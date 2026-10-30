"""APScheduler 采集调度器。

运行方式：
    python -m crawler.scheduler --once
    python -m crawler.scheduler
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from sqlalchemy import text

from app.config import settings
from app.db import SessionLocal, init_db
from app.models import Keyword, Post, WatchAccount
from app.seed import ensure_admin, seed_demo
from app.services.worker import process_queue

from .client import MockWeiboClient, RotatingWeiboClient
from .tasks import CrawlService

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def build_client(db):
    if settings.crawler_mode == "mock":
        return MockWeiboClient()
    return RotatingWeiboClient(db)


def run_cycle() -> dict[str, int]:
    db = SessionLocal()
    try:
        service = CrawlService(db, build_client(db))
        result = service.run_all_once()
        # 采集后顺手消费一小批情感队列，避免 worker 未单独启动时数据一直未打分。
        result["sentiment"] = process_queue(db, batch_size=32, max_batches=3)
        logger.info("采集轮次完成：%s", result)
        return result
    finally:
        db.close()


def job_keywords() -> None:
    db = SessionLocal()
    try:
        service = CrawlService(db, build_client(db))
        now = datetime.utcnow()
        for kw in db.query(Keyword).filter(Keyword.enabled.is_(True)).all():
            if kw.expire_at and kw.expire_at < now:
                continue
            try:
                service.run_keyword(kw.word)
            except Exception as exc:
                logger.warning("关键词任务失败 %s：%s", kw.word, exc)
    finally:
        db.close()


def job_accounts() -> None:
    db = SessionLocal()
    try:
        service = CrawlService(db, build_client(db))
        for acc in db.query(WatchAccount).filter(WatchAccount.enabled.is_(True)).all():
            try:
                service.run_account(acc.uid)
            except Exception as exc:
                logger.warning("账号任务失败 %s：%s", acc.uid, exc)
    finally:
        db.close()


def job_hotsearch() -> None:
    db = SessionLocal()
    try:
        CrawlService(db, build_client(db)).run_hotsearch()
    except Exception as exc:
        logger.warning("热搜任务失败：%s", exc)
    finally:
        db.close()


def job_interactions() -> None:
    db = SessionLocal()
    try:
        service = CrawlService(db, build_client(db))
        since = datetime.utcnow() - timedelta(hours=24)
        rows = (
            db.query(Post)
            .filter(Post.created_at >= since)
            .order_by(Post.reposts.desc())
            .limit(200)
            .all()
        )
        for post in rows:
            try:
                service.run_comments(post.mid)
                service.run_reposts(post.mid)
            except Exception as exc:
                logger.warning("互动任务失败 %s：%s", post.mid, exc)
    finally:
        db.close()


def job_sentiment() -> None:
    db = SessionLocal()
    try:
        process_queue(db, batch_size=32, max_batches=3)
    finally:
        db.close()


def job_cleanup() -> None:
    """清理过期热搜/临时关键词与超过保留期的原始 JSON。"""
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        db.query(Keyword).filter(Keyword.is_temp.is_(True), Keyword.expire_at < now).delete(
            synchronize_session=False
        )
        db.execute(
            text(
                "UPDATE post SET raw = NULL WHERE fetched_at < :cutoff"
            ),
            {"cutoff": now - timedelta(days=settings.raw_retention_days)},
        )
        db.commit()
    finally:
        db.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="微博采集调度器")
    ap.add_argument("--once", action="store_true", help="只跑一轮后退出")
    ap.add_argument("--no-seed", action="store_true")
    args = ap.parse_args()

    init_db()
    db = SessionLocal()
    try:
        ensure_admin(db)
        if settings.seed_demo and not args.no_seed:
            seed_demo(db, settings.demo_csv)
    finally:
        db.close()

    if args.once:
        print(run_cycle())
        return 0

    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    common = {"max_instances": 1, "coalesce": True}
    scheduler.add_job(job_keywords, "interval", minutes=15, id="keywords", **common)
    scheduler.add_job(job_accounts, "interval", minutes=30, id="accounts", **common)
    scheduler.add_job(job_hotsearch, "interval", minutes=10, id="hotsearch", **common)
    scheduler.add_job(job_interactions, "interval", hours=2, id="interactions", **common)
    scheduler.add_job(job_sentiment, "interval", seconds=30, id="sentiment", **common)
    scheduler.add_job(job_cleanup, "cron", hour=3, minute=30, id="cleanup", **common)
    logger.info("调度器启动：mode=%s，jobs=%s", settings.crawler_mode, [j.id for j in scheduler.get_jobs()])
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("调度器退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
