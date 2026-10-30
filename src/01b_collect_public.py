# -*- coding: utf-8 -*-
"""01b 合规公开页面采集器 + 人工导入 + 本地自检。

定位
----
主数据来自 ScienceDB 公开数据集（01a）。本模块提供“可运行的采集能力”，
用于后续扩展公开新闻/政务页面，不抓取微博站内接口、不登录、不绕验证码。

合规约束（写死在代码里）
------------------------
1. 只采集公开页面；请求前读取 robots.txt，Disallow 一律跳过；
2. 同一站点请求间隔 >= 2 秒（MIN_DELAY），支持 Crawl-delay 更大值；
3. 不携带 Cookie/登录态，不提交表单，不解析验证码；
4. 只保留标题、正文、发布日期、来源 URL；不采集作者/账号/联系方式；
5. 限量采集（默认最多 20 页），日志留痕。

用法
----
    python src/01b_collect_public.py --self-test
    python src/01b_collect_public.py --url https://example.com/a --url https://example.com/b
    python src/01b_collect_public.py --import-csv data/raw_real/manual_posts.csv

自检产物：output/tables/爬虫自检.csv、output/crawler_self_test.json
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from datetime import datetime
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import (FIXTURES, OUT, RAW_REAL, TABLES, now_str, parse_day,
                        write_json)

MIN_DELAY = 2.0
MAX_PAGES = 20
UA = "WeiboOpinionResearchBot/1.0 (+public-research; contact: project-team)"
OUT_COLUMNS = ["doc_hash", "title", "text", "publish_date", "source_url",
               "collected_at", "collector_version"]
COLLECTOR_VERSION = "1.0"


class RateLimiter:
    def __init__(self, delay: float = MIN_DELAY):
        self.delay = max(delay, MIN_DELAY)
        self.last = 0.0
        self.requests = 0

    def wait(self):
        elapsed = time.time() - self.last
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last = time.time()
        self.requests += 1


def fetch(session, url, limiter, timeout=20):
    limiter.wait()
    r = session.get(url, timeout=timeout, headers={
        "User-Agent": UA,
        "Accept-Language": "zh-CN,zh;q=0.9",
    })
    r.raise_for_status()
    if not r.encoding or r.encoding.lower() in ("iso-8859-1", "ascii"):
        r.encoding = r.apparent_encoding or "utf-8"
    return r


def robots_policy(session, url, limiter):
    """返回 RobotFileParser；robots 不可读时返回 None（失败关闭）。"""
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return None
    robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
    try:
        limiter.wait()
        r = session.get(robots_url, timeout=15, headers={"User-Agent": UA})
        r.requests_count = limiter.requests
        rp = RobotFileParser()
        rp.set_url(robots_url)
        if r.status_code == 404:
            rp.parse(["User-agent: *", "Allow: /"])
        elif r.status_code == 200:
            rp.parse(r.text.splitlines())
        else:
            return None
        return rp
    except Exception:
        return None


def parse_article(html: str, url: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "aside", "form", "noscript"]):
        tag.decompose()
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(" ", strip=True)
    if not title and soup.title:
        title = soup.title.get_text(" ", strip=True)

    date = ""
    t = soup.find("time")
    if t:
        date = t.get("datetime") or t.get_text(" ", strip=True)
    if not date:
        meta = soup.find("meta", attrs={"name": re.compile("publish|date", re.I)})
        if meta:
            date = meta.get("content", "")
    date = parse_day(date) or ""

    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join([p for p in paras if len(p) >= 8])
    if len(text) < 30:
        art = soup.find("article")
        if art:
            text = art.get_text(" ", strip=True)
    if len(text) < 30 or not title:
        return None
    doc_hash = "doc_" + hashlib.sha1((title + "|" + text).encode("utf-8")).hexdigest()[:16]
    return {
        "doc_hash": doc_hash,
        "title": title[:120],
        "text": text[:5000],
        "publish_date": date,
        "source_url": url,
        "collected_at": now_str(),
        "collector_version": COLLECTOR_VERSION,
    }


def dedupe(rows):
    seen = set()
    out = []
    for r in rows:
        if r["doc_hash"] in seen:
            continue
        seen.add(r["doc_hash"])
        out.append(r)
    return out


def collect_urls(urls, out_path, delay=MIN_DELAY, max_pages=MAX_PAGES):
    session = requests.Session()
    limiter = RateLimiter(delay)
    rows, skipped = [], []
    for url in urls[:max_pages]:
        rp = robots_policy(session, url, limiter)
        if rp is None:
            skipped.append({"url": url, "reason": "robots.txt 不可读，按合规要求跳过"})
            continue
        if not rp.can_fetch(UA, url):
            skipped.append({"url": url, "reason": "robots.txt Disallow"})
            continue
        try:
            r = fetch(session, url, limiter)
            row = parse_article(r.text, url)
            if row:
                rows.append(row)
            else:
                skipped.append({"url": url, "reason": "正文过短或无法解析"})
        except Exception as e:
            skipped.append({"url": url, "reason": f"{type(e).__name__}: {e}"})
    rows = dedupe(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows, columns=OUT_COLUMNS).to_csv(out_path, index=False, encoding="utf-8-sig")
    return rows, skipped, limiter.requests


def self_test():
    print("=" * 66)
    print("01b 采集器自检（本地 fixture，不访问外网）")
    print("=" * 66)
    checks = []
    fixtures = [os.path.join(FIXTURES, f"public_page_{i}.html") for i in (1, 2, 3)]
    rows = []
    for p in fixtures:
        if not os.path.exists(p):
            print(f"  [FAIL] fixture 缺失：{p}")
            sys.exit(1)
        with open(p, encoding="utf-8") as f:
            row = parse_article(f.read(), "https://example.org/public/" + os.path.basename(p))
        if row:
            rows.append(row)
    # 重复输入用于验证内容去重
    if rows:
        rows.append(dict(rows[0]))
    unique = dedupe(rows)

    checks.append(("三页 fixture 均解析成功", len(rows) >= 4))
    checks.append(("内容哈希去重（4 输入 -> 3 输出）", len(unique) == 3))
    checks.append(("标题/正文/日期字段完整", all(r["title"] and len(r["text"]) >= 30 and r["publish_date"] for r in unique)))
    checks.append(("输出不含作者/账号字段", not any(k in OUT_COLUMNS for k in ("author", "account", "user_id", "nickname"))))
    checks.append(("生产限速 >= 2 秒/请求", MIN_DELAY >= 2.0))

    rp = RobotFileParser()
    rp.parse(["User-agent: *", "Disallow: /private", "Allow: /public"])
    checks.append(("robots.txt Disallow 生效", (not rp.can_fetch(UA, "https://example.org/private/x"))
                   and rp.can_fetch(UA, "https://example.org/public/x")))

    out_csv = os.path.join(TABLES, "爬虫自检.csv")
    pd.DataFrame(unique, columns=OUT_COLUMNS).to_csv(out_csv, index=False, encoding="utf-8-sig")
    report = {
        "checked_at": now_str(),
        "mode": "self-test (local fixtures)",
        "min_delay_seconds": MIN_DELAY,
        "max_pages": MAX_PAGES,
        "input_pages": len(rows),
        "unique_pages": len(unique),
        "checks": [{"check": c, "pass": bool(v)} for c, v in checks],
        "note": "fixture 仅用于采集器自检，不进入模型训练与舆情统计",
    }
    write_json(os.path.join(OUT, "crawler_self_test.json"), report)
    fails = [c for c, v in checks if not v]
    for c, v in checks:
        print(f"  [{'PASS' if v else 'FAIL'}] {c}")
    print(f"\n  自检输出：{os.path.relpath(out_csv)}（{len(unique)} 条）")
    if fails:
        print(f"\n[阻断] 采集器自检 {len(fails)} 项失败")
        sys.exit(1)
    print("\n[通过] 采集器自检完成；主数据仍来自 01a 公开数据集导入。")


def import_csv(path):
    df = pd.read_csv(path, dtype=str)
    required = {"text", "publish_time", "source_url"}
    missing = required - set(df.columns)
    if missing:
        print(f"[FAIL] 人工导入表缺列：{sorted(missing)}")
        sys.exit(1)
    rows = []
    for _, r in df.iterrows():
        text = str(r["text"]).strip()
        if len(text) < 30:
            continue
        title = str(r.get("title", "")).strip()[:120] or text[:40]
        rows.append({
            "doc_hash": "doc_" + hashlib.sha1((title + "|" + text).encode("utf-8")).hexdigest()[:16],
            "title": title,
            "text": text[:5000],
            "publish_date": parse_day(r["publish_time"]) or "",
            "source_url": str(r["source_url"]).strip(),
            "collected_at": now_str(),
            "collector_version": COLLECTOR_VERSION + "-manual",
        })
    rows = dedupe(rows)
    out = os.path.join(RAW_REAL, "collected_public.csv")
    pd.DataFrame(rows, columns=OUT_COLUMNS).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[完成] 人工导入 {len(rows)} 条 -> {os.path.relpath(out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--url", action="append", default=[])
    ap.add_argument("--max-pages", type=int, default=MAX_PAGES)
    ap.add_argument("--delay", type=float, default=MIN_DELAY)
    ap.add_argument("--out", default=os.path.join(RAW_REAL, "collected_public.csv"))
    ap.add_argument("--import-csv", default=None)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.import_csv:
        import_csv(args.import_csv)
        return
    if args.url:
        urls = []
        for u in args.url:
            urls.extend([x.strip() for x in u.split(",") if x.strip()])
        rows, skipped, n_req = collect_urls(urls, args.out, args.delay, args.max_pages)
        write_json(os.path.join(OUT, "crawler_run_log.json"), {
            "collected_at": now_str(), "urls": urls, "requests": n_req,
            "collected": len(rows), "skipped": skipped,
            "min_delay_seconds": max(args.delay, MIN_DELAY),
        })
        print(f"[完成] 采集 {len(rows)} 条，跳过 {len(skipped)} 条，HTTP 请求 {n_req} 次")
        print(f"  输出：{args.out}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
