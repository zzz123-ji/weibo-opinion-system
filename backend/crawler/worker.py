"""情感 worker 命令行。

    python -m crawler.worker --once
    python -m crawler.worker --loop --sleep 5
"""

from __future__ import annotations

import argparse
import logging
import time

from app.config import settings
from app.db import SessionLocal, init_db
from app.services.queue import queue_counts
from app.services.worker import process_queue

logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description="情感打分 worker")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--version", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-batches", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=5.0)
    args = ap.parse_args()

    init_db()
    if args.once or not args.loop:
        db = SessionLocal()
        try:
            n = process_queue(
                db, args.engine, args.version, args.batch_size, args.max_batches
            )
            logger.info("处理 %s 条；队列状态 %s", n, queue_counts(db))
        finally:
            db.close()
        return 0

    logger.info("worker 循环启动，sleep=%.1fs", args.sleep)
    while True:
        db = SessionLocal()
        try:
            n = process_queue(
                db, args.engine, args.version, args.batch_size, args.max_batches
            )
            if n:
                logger.info("处理 %s 条；队列 %s", n, queue_counts(db))
        except Exception as exc:  # pragma: no cover - 守守护进程
            logger.exception("worker 异常：%s", exc)
        finally:
            db.close()
        time.sleep(args.sleep)


if __name__ == "__main__":
    raise SystemExit(main())
