from __future__ import annotations

from datetime import datetime

from tests.conftest import add_post


def test_health_and_auth(client):
    assert client.get("/api/health").status_code == 200
    bad = client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})
    assert bad.status_code == 401
    resp = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["role"] == "admin"


def test_permissions(client, annotator_headers):
    assert client.get("/api/config/keywords", headers=annotator_headers).status_code == 200
    denied = client.post(
        "/api/config/keywords", headers=annotator_headers, json={"word": "测试"}
    )
    assert denied.status_code == 403


def test_posts_and_trends(client, admin_headers):
    add_post("m1", "武汉加油，感谢医护人员", datetime(2020, 1, 20, 10), label="pos", reposts=100)
    add_post("m2", "口罩买不到，太失望了", datetime(2020, 1, 21, 11), label="neg", reposts=20, source="口罩")
    add_post("m3", "今天天气不错", datetime(2020, 1, 21, 12), label="neu", reposts=1, source="其他")

    posts = client.get("/api/posts?page_size=10", headers=admin_headers)
    assert posts.status_code == 200, posts.text
    assert posts.json()["total"] == 3

    filtered = client.get("/api/posts?sentiment=neg", headers=admin_headers)
    assert filtered.status_code == 200
    assert [x["mid"] for x in filtered.json()["items"]] == ["m2"]

    by_kw = client.get("/api/posts?keyword=武汉", headers=admin_headers)
    assert by_kw.status_code == 200
    assert [x["mid"] for x in by_kw.json()["items"]] == ["m1"]

    volume = client.get("/api/trend/volume?granularity=day", headers=admin_headers)
    assert volume.status_code == 200
    assert sum(x["count"] for x in volume.json()["points"]) == 3

    sentiment = client.get("/api/trend/sentiment?granularity=day", headers=admin_headers)
    assert sentiment.status_code == 200
    assert sentiment.json()["points"]

    top = client.get("/api/trend/top-negative?limit=5", headers=admin_headers)
    assert top.status_code == 200, top.text
    assert top.json()[0]["mid"] == "m2"
    assert top.json()[0]["sentiment"] == "neg"

    detail = client.get("/api/posts/m2", headers=admin_headers)
    assert detail.status_code == 200
    assert detail.json()["sentiment"] == "neg"
    assert detail.json()["interactions"] == 20 + 2 + 5


def test_label_flow_and_final_view(client, admin_headers, annotator_headers):
    add_post("m1", "普通文本", datetime(2020, 1, 1), label="neg")

    nxt = client.get("/api/label/next", headers=annotator_headers)
    assert nxt.status_code == 200, nxt.text
    assert nxt.json()["text_id"] == "m1"

    put = client.put(
        "/api/label/post/m1", headers=annotator_headers, json={"label": "pos"}
    )
    assert put.status_code == 200, put.text
    assert put.json()["final_label"] == "pos"
    assert put.json()["final_source"] == "manual"

    detail = client.get("/api/posts/m1", headers=admin_headers)
    assert detail.json()["sentiment"] == "pos"
    assert detail.json()["sentiment_source"] == "manual"

    stats = client.get("/api/label/stats", headers=annotator_headers)
    assert stats.status_code == 200
    assert stats.json()["my_total"] == 1

    # 已标注样本不会再出现在同一个人的队列里。
    again = client.get("/api/label/next", headers=annotator_headers)
    assert again.status_code == 404

    export = client.get("/api/label/export", headers=admin_headers)
    assert export.status_code == 200
    assert "普通文本" in export.text


def test_final_view_tie_uses_latest(client, admin_headers, annotator_headers):
    add_post("m1", "打平测试", datetime(2020, 1, 1), label="neg")
    client.put("/api/label/post/m1", headers=annotator_headers, json={"label": "neg"})
    # 第二个标注员给 pos，票数打平，最新一条是 pos。
    second = client.post(
        "/api/config/users",
        headers=admin_headers,
        json={"username": "annotator2", "password": "pass1234", "role": "annotator"},
    )
    assert second.status_code == 200
    h2 = client.post(
        "/api/auth/login", json={"username": "annotator2", "password": "pass1234"}
    ).json()
    h2 = {"Authorization": f"Bearer {h2['access_token']}"}
    client.put("/api/label/post/m1", headers=h2, json={"label": "pos"})
    detail = client.get("/api/posts/m1", headers=admin_headers)
    assert detail.json()["sentiment"] == "pos"


def test_eval(client, admin_headers, annotator_headers):
    add_post("m1", "正面文本", datetime(2020, 1, 1), label="pos", score=0.9)
    add_post("m2", "负面文本", datetime(2020, 1, 2), label="neg", score=0.8)
    client.put("/api/label/post/m1", headers=annotator_headers, json={"label": "pos"})
    client.put("/api/label/post/m2", headers=annotator_headers, json={"label": "neg"})
    resp = client.get("/api/eval", headers=admin_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["n_manual_texts"] == 2
    assert body["engines"][0]["accuracy"] == 1.0
    assert body["engines"][0]["engine"] == "mock"


def test_config_crud(client, admin_headers):
    created = client.post(
        "/api/config/keywords", headers=admin_headers, json={"word": "新关键词"}
    )
    assert created.status_code == 200, created.text
    kid = created.json()["id"]
    updated = client.put(
        f"/api/config/keywords/{kid}", headers=admin_headers, json={"enabled": False}
    )
    assert updated.status_code == 200
    assert updated.json()["enabled"] is False
    deleted = client.delete(f"/api/config/keywords/{kid}", headers=admin_headers)
    assert deleted.status_code == 200

    cookie = client.post(
        "/api/config/cookies",
        headers=admin_headers,
        json={"account_name": "demo", "cookie": "SUB=abcdefghijklmnopqrstuvwxyz"},
    )
    assert cookie.status_code == 200
    assert "cookie_enc" not in cookie.json()


def test_engine_switch_and_rerun(client, admin_headers):
    add_post("m1", "测试", datetime(2020, 1, 1), label="neg")
    switched = client.post(
        "/api/config/engine", headers=admin_headers, json={"name": "mock", "version": "test"}
    )
    assert switched.status_code == 200
    rerun = client.post(
        "/api/sentiment/rerun",
        headers=admin_headers,
        json={"engine": "mock", "version": "test", "limit": 10},
    )
    assert rerun.status_code == 200, rerun.text
    run = client.post("/api/sentiment/run-once", headers=admin_headers)
    assert run.status_code == 200, run.text
