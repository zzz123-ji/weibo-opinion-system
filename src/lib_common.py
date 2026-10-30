# -*- coding: utf-8 -*-
"""公共工具：路径、JSON、哈希、日期、标签口径、控制台编码。

所有脚本统一从本模块取路径，保证 run_all.py 与单脚本运行结果一致。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime

# PowerShell 控制台默认 GBK，统一改为 UTF-8，避免中文/特殊字符崩溃
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = HERE
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
PROCESSED = os.path.join(DATA, "processed")
FIXTURES = os.path.join(DATA, "fixtures")
RAW_REAL = os.path.join(DATA, "raw_real")
OUT = os.path.join(ROOT, "output")
TABLES = os.path.join(OUT, "tables")
MODELS = os.path.join(OUT, "models")
DOCS = os.path.join(ROOT, "docs")
FIGURES = os.path.join(DOCS, "figures")
TEMPLATES = os.path.join(DOCS, "templates")
DASHBOARD = os.path.join(ROOT, "dashboard")
DASH_ASSETS = os.path.join(DASHBOARD, "assets")
PDF_DIR = os.path.join(DOCS, "pdf")

for _p in (DATA, RAW, PROCESSED, FIXTURES, RAW_REAL, OUT, TABLES, MODELS,
           DOCS, FIGURES, TEMPLATES, DASHBOARD, DASH_ASSETS, PDF_DIR):
    os.makedirs(_p, exist_ok=True)

# 标签口径：-1 负面 / 0 中性 / 1 正面（ScienceDB 数据集说明）
LABEL_NAMES = {-1: "负面", 0: "中性", 1: "正面"}
LABEL_ORDER = [-1, 0, 1]
LABEL_COLORS = {-1: "#ff5d73", 0: "#8fa3c8", 1: "#2ee6a8"}

DATASET_DOI = "10.57760/sciencedb.j00133.00366"
DATASET_URL = "https://doi.org/" + DATASET_DOI
DATASET_TITLE = "Weibo and Twitter datasets（微博图文情感子集）"
DATASET_LICENSE = "CC BY 4.0"
DATASET_PUBLISHER = "Science Data Bank (ScienceDB)"

RANDOM_SEED = 42
SAMPLE_PER_CLASS = 4000
EXTERNAL_PER_CLASS = 1000
MIN_TEXT_LEN = 4


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def post_hash(raw_id: str, text: str = "") -> str:
    """用不可逆哈希替代原始微博 ID，保留跨表关联能力。"""
    h = hashlib.sha256((str(raw_id) + "|" + str(text)[:80]).encode("utf-8"))
    return "wb_" + h.hexdigest()[:16]


def parse_day(s) -> str | None:
    """'01月23日 12:30' -> '2020-01-23'；兼容 '2020-01-23'。"""
    if s is None:
        return None
    s = str(s)
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(\d{1,2})月(\d{1,2})日", s)
    if m:
        return f"2020-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return None


def fmt_int(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)


def fmt_pct(x, nd: int = 2) -> str:
    try:
        return f"{float(x) * 100:.{nd}f}%"
    except Exception:
        return str(x)


def fmt_f(x, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace("\\", "/")


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)
