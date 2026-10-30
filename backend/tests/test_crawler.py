from __future__ import annotations

from datetime import datetime, timedelta

from app.db import SessionLocal
from app.models import Keyword, Post, SentimentQueue, WeiboCookie
from app.security import encrypt_cookie
from crawler.client import AccountPool, MockWeiboClient
from crawler.parsers import parse_comments, parse_hotsearch, parse_reposts, parse_search
from crawler.tasks import CrawlService


def test_parsers_with_fixtures():
    client = MockWeiboClient()
    posts = client.search_keyword("武汉")
    assert len(posts) == 3
    assert posts[0]["mid"] == "5000000000000001"
    assert "武汉加油" in posts[0]["text"]
    assert posts[0]["author"]["uid"] == "10001"

    comments = client.comments("5000000000000001")
    assert len(comments) == 2
    assert comments[0]["mid"] == "5000000000000001"

    reposts = client.reposts("5000000000000001")
    assert len(reposts) == 2
    assert reposts[0]["root_mid"] == "5000000000000001"

    hot = client.hot_search()
    assert len(hot) == 5
    assert hot[0]["rank"] == 1


def test_account_pool_cooldown_and_switch():
    db = SessionLocal()
    try:
        cookie = WeiboCookie(
            account_name="a1",
            cookie_enc=encrypt_cookie("SUB=test-cookie"),
            status="cooling",
            cooling_until=datetime.utcnow() - timedelta(minutes=1),
            fail_count=1,
        )
        db.add(cookie)
        db.commit()
        pool = AccountPool(db, min_interval=0, sleeper=lambda _x: None)
        got = pool.acquire()
        assert got.status == "ok"
        pool.report_rate_limited(got)
        db.refresh(got)
        assert got.status == "cooling"
        assert got.cooling_until is not None
        for _ in range(2):
            got.status = "ok"
            got.cooling_until = None
            db.commit()
            pool.report_rate_limited(got)
            db.refresh(got)
        assert got.status == "abnormal"
    finally:
        db.close()


def test_mock_crawl_service_inserts_and_dedups():
    db = SessionLocal()
    try:
        db.add(Keyword(word="武汉", enabled=True))
        db.commit()
        service = CrawlService(db, MockWeiboClient(), max_pages=1)
        first = service.run_keyword("武汉")
        assert first == 3
        assert db.query(Post).count() == 3
        assert db.query(SentimentQueue).count() == 3
        second = service.run_keyword("武汉")
        assert second == 0
        assert db.query(Post).count() == 3
    finally:
        db.close()
