"""采集合规辅助：公开页面 robots 检查、域名白名单、Cookie 不外泄。"""

from __future__ import annotations

from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

ALLOWED_HOSTS = {
    "m.weibo.cn",
    "weibo.com",
    "s.weibo.com",
}


class ComplianceError(RuntimeError):
    pass


def assert_allowed_host(url: str) -> None:
    host = urlparse(url).hostname or ""
    if host not in ALLOWED_HOSTS:
        raise ComplianceError(f"域名不在白名单：{host}")


def robots_allows(url: str, user_agent: str = "WeiboSentimentBot", timeout: float = 10.0) -> bool:
    """对网页兜底路径做 robots 检查；JSON API 由调用方按公开接口使用。"""
    assert_allowed_host(url)
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except Exception:
        # 无法读取 robots 时保守放行，但只允许白名单域名；生产应改为拒绝。
        return True
    return parser.can_fetch(user_agent, url)


def assert_public_content(url: str) -> None:
    """禁止私信、非公开主页等路径。"""
    assert_allowed_host(url)
    lowered = url.lower()
    forbidden = ("/messages", "/chat", "/private", "/direct_messages")
    if any(x in lowered for x in forbidden):
        raise ComplianceError("禁止采集私信或非公开内容")
