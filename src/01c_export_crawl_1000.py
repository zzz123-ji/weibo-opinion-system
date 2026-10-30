# -*- coding: utf-8 -*-
"""01c 采集样本落盘：1,000 条“user_id + time + context(文本+图片)”数据。

数据来源
--------
ScienceDB 公开数据集（DOI: 10.57760/sciencedb.j00133.00366）：
- nCoV_100k_train_labled.csv：发布时间、发布账号、正文、图片 URL、情感标签
- image_text_label.csv：图文主表（提供 zip 内图片文件名与标签）
- dataset.zip：图片文件

为什么不是“实时爬微博”
----------------------
微博站内接口/用户页需要登录且返回 403，按项目红线不绕验证码、不携带登录态。
本脚本把公开数据集中的 1,000 条样本按“采集落盘”规范导出，图片从官方 zip 精确抽取；
另提供 --download-images 选项，可从公开图片 CDN 逐张下载（>=2 秒/请求）。

隐私处理
--------
- 原始账号名只在内存中生成不可逆 user_id（sha256 前 16 位），默认不落盘；
- 如确需内部核对，可加 --keep-raw-account，生成 data/crawl/_internal_account_map.csv，
  该文件仅供内部排查，不得提交；
- 输出表不含原始微博 ID、图片 URL、账号名；这些只放在 _internal_provenance.csv。

用法
----
    python src/01c_export_crawl_1000.py --n 1000
    python src/01c_export_crawl_1000.py --n 1000 --keep-raw-account
    python src/01c_export_crawl_1000.py --n 1000 --download-images   # zip 缺图时
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import (DATASET_DOI, DATASET_URL, LABEL_NAMES, RAW, ROOT,
                        now_str, parse_day, post_hash, write_json)
from lib_text import clean_text, normalize_for_match, valid_text

CRAWL_DIR = os.path.join(ROOT, "data", "crawl")
IMG_DIR = os.path.join(CRAWL_DIR, "images")
ZIP_IMG_PREFIX = "dataset/Weibo/train_dataset/image/"
UA = "WeiboOpinionResearchBot/1.0 (+public-research)"


def read_csv_smart(path, **kw):
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kw)
    except Exception:
        return pd.read_csv(path, encoding="gb18030", encoding_errors="replace", **kw)


def parse_datetime(s) -> str:
    """'01月03日 23:02' -> '2020-01-03 23:02:00'。"""
    if s is None:
        return ""
    s = str(s).strip()
    m = re.search(r"(\d{1,2})月(\d{1,2})日\s*(\d{1,2}):(\d{2})", s)
    if m:
        return f"2020-{int(m.group(1)):02d}-{int(m.group(2)):02d} {int(m.group(3)):02d}:{m.group(4)}:00"
    d = parse_day(s)
    return (d + " 00:00:00") if d else ""


def user_hash(account: str, text: str = "") -> str:
    raw = str(account).strip()
    if not raw:
        raw = "unknown|" + str(text)[:40]
    return "u_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def find_zip(arg_zip=None):
    candidates = [
        arg_zip,
        os.path.join(RAW, "dataset.zip"),
        os.environ.get("WEIBO_ZIP", ""),
        r"C:\Users\zzz13\AppData\Local\Temp\opencode\weibo_dataset.zip",
    ]
    for p in candidates:
        if p and os.path.exists(p) and os.path.getsize(p) > 1024 * 1024:
            return p
    return None


def build_candidates():
    main = read_csv_smart(os.path.join(RAW, "image_text_label.csv"), dtype=str)
    main.columns = ["weibo_id", "text", "image_url", "label"][:len(main.columns)]
    main["key"] = main["text"].map(normalize_for_match)
    main = main[main["key"].str.len() > 0].drop_duplicates("key", keep="first")
    lookup = main.set_index("key")[["weibo_id", "image_url", "label"]]

    nc = read_csv_smart(os.path.join(RAW, "nCoV_100k_train_labled.csv"), dtype=str)
    nc.columns = ["weibo_id", "publish_time", "account", "text", "images",
                  "videos", "label"][:len(nc.columns)]
    nc["label_num"] = pd.to_numeric(nc["label"], errors="coerce")
    nc = nc[nc["label_num"].isin([-1, 0, 1])].copy()
    nc["label"] = nc["label_num"].astype(int)
    nc["n_img"] = nc["images"].fillna("[]").map(
        lambda s: len(re.findall(r"https?://[^\s'\"\],]+", s)))
    nc["key"] = nc["text"].map(normalize_for_match)
    nc["source_row_100k"] = nc.index.astype(int)
    nc = nc[(nc["n_img"] > 0) & (nc["key"].str.len() > 0)
            & (nc["text"].map(valid_text))].drop_duplicates("key", keep="first")
    m = nc.merge(lookup, left_on="key", right_index=True, how="inner",
                 suffixes=("_nc", "_main"))
    m["label"] = pd.to_numeric(m["label_main"], errors="coerce").astype("Int64")
    m = m[m["label"].isin([-1, 0, 1])].copy()
    m["label"] = m["label"].astype(int)
    return m


def sample_balanced(df, n):
    parts = []
    base = n // 3
    remainder = n - base * 3
    for i, lab in enumerate((-1, 0, 1)):
        want = base + (1 if i < remainder else 0)
        sub = df[df["label"] == lab]
        take = min(want, len(sub))
        if take < want:
            print(f"  [WARN] 类别 {lab} 仅 {len(sub):,} 条，取 {take:,}")
        parts.append(sub.sample(n=take, random_state=42 + i))
    out = pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return out


def extract_images(zf, rows, out_dir, force=False):
    os.makedirs(out_dir, exist_ok=True)
    names = set(zf.namelist())
    rows_out, missing = [], []
    for _, r in rows.iterrows():
        weibo_id = str(r["weibo_id_main"])
        cand = [f"{ZIP_IMG_PREFIX}{weibo_id}.jpg",
                f"{ZIP_IMG_PREFIX}{weibo_id}.jpeg",
                f"{ZIP_IMG_PREFIX}{weibo_id}.png"]
        name = next((c for c in cand if c in names), None)
        if name is None:
            missing.append(r)
            continue
        ext = os.path.splitext(name)[1].lower()
        dest = os.path.join(out_dir, str(r["post_id"]) + ext)
        if force or not (os.path.exists(dest) and os.path.getsize(dest) > 0):
            with zf.open(name) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
        rows_out.append((r, dest, name))
    return rows_out, missing


def download_missing(rows, out_dir):
    import requests

    session = requests.Session()
    last = 0.0
    ok = 0
    for r in rows:
        url = str(r["image_url"])
        if not url.startswith("http"):
            continue
        elapsed = time.time() - last
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)
        try:
            resp = session.get(url, timeout=20, headers={"User-Agent": UA})
            last = time.time()
            resp.raise_for_status()
            ext = ".jpg"
            dest = os.path.join(out_dir, str(r["post_id"]) + ext)
            with open(dest, "wb") as f:
                f.write(resp.content)
            ok += 1
        except Exception:
            last = time.time()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--zip", default=None)
    ap.add_argument("--keep-raw-account", action="store_true",
                    help="生成内部账号映射（不得提交）")
    ap.add_argument("--download-images", action="store_true",
                    help="zip 缺图时从公开图片 CDN 下载（>=2 秒/请求）")
    args = ap.parse_args()

    os.makedirs(CRAWL_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    print("=" * 66)
    print(f"01c 采集样本落盘：{args.n:,} 条 user_id + time + 文本 + 图片")
    print("=" * 66)
    print(f"  数据来源：{DATASET_URL}（CC BY 4.0）")

    cand = build_candidates()
    print(f"  候选（有图 + 有日期 + 文本可匹配） {len(cand):,} 条；"
          f"类别 {cand['label'].value_counts().sort_index().to_dict()}；"
          f"账号 {cand['account'].nunique():,} 个")
    picked = sample_balanced(cand, args.n)
    picked["post_id"] = [post_hash(r["weibo_id_main"], r["text"]) for _, r in picked.iterrows()]
    picked["user_id"] = [user_hash(r["account"], r["text"]) for _, r in picked.iterrows()]
    print(f"  抽样 {len(picked):,} 条；类别 {picked['label'].value_counts().sort_index().to_dict()}；"
          f"唯一 user_id {picked['user_id'].nunique():,} 个")

    zip_path = find_zip(args.zip)
    rows_out, missing = [], []
    if zip_path:
        print(f"  从官方 zip 抽取图片：{zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            rows_out, missing = extract_images(zf, picked, IMG_DIR)
    else:
        missing = [r for _, r in picked.iterrows()]
        print("  [WARN] 未找到 dataset.zip")

    if missing and args.download_images:
        print(f"  zip 缺图 {len(missing):,} 条，尝试从公开 CDN 下载（>=2 秒/请求）")
        md = pd.DataFrame(missing)
        ok = download_missing(md, IMG_DIR)
        print(f"  下载成功 {ok:,} 条")
        # 重新扫描
        rows_out = []
        for _, r in picked.iterrows():
            for ext in (".jpg", ".jpeg", ".png"):
                p = os.path.join(IMG_DIR, str(r["post_id"]) + ext)
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    rows_out.append((r, p, ""))
                    break

    # 组装落盘表
    records = []
    for r, path, zip_name in rows_out:
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            w, h = 0, 0
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()[:16]
        text_full = clean_text(r["text"])
        text_clean = text_full[:1000]
        image_rel = os.path.relpath(path, CRAWL_DIR).replace("\\", "/")
        context = json.dumps({"text": text_clean, "image": image_rel},
                             ensure_ascii=False, separators=(",", ":"))
        records.append({
            "post_id": r["post_id"],
            "user_id": r["user_id"],
            "user_id_type": "sha256_pseudonym",
            "time": parse_datetime(r["publish_time"]),
            "date": parse_day(r["publish_time"]),
            "text": text_clean,
            "text_len": len(text_full),
            "label": int(r["label"]),
            "label_name": LABEL_NAMES[int(r["label"])],
            "image_file": image_rel,
            "context": context,
            "image_sha256": sha,
            "image_width": w,
            "image_height": h,
            "image_bytes": os.path.getsize(path),
            "source": "ScienceDB Weibo train_dataset（公开图文子集）",
            "source_type": "public_dataset_import",
            "dataset_doi": DATASET_DOI,
            "crawl_time": now_str(),
        })
    df = pd.DataFrame(records)
    out_csv = os.path.join(CRAWL_DIR, "weibo_1000.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 内部溯源：不含账号名；账号名仅在 --keep-raw-account 时单独写
    prov = picked[["post_id", "user_id", "weibo_id_main", "weibo_id_nc",
                   "image_url", "publish_time", "label", "source_row_100k"]].copy()
    prov["dataset_doi"] = DATASET_DOI
    prov.to_csv(os.path.join(CRAWL_DIR, "_internal_provenance.csv"),
                index=False, encoding="utf-8-sig")
    if args.keep_raw_account:
        picked[["post_id", "user_id", "account"]].to_csv(
            os.path.join(CRAWL_DIR, "_internal_account_map.csv"),
            index=False, encoding="utf-8-sig")
        print("  [注意] 已生成 _internal_account_map.csv（含原始账号，禁止提交）")

    # 说明文档
    readme = f"""# 1,000 条采集样本（user_id + time + context）

## 这是什么
本目录是“公开数据集采集落盘”演示样本，共 {len(df):,} 条，来自 ScienceDB 公开数据集
《Weibo and Twitter datasets》微博图文子集（DOI: {DATASET_DOI}，CC BY 4.0）。
不是实时爬取微博站内接口：微博接口/用户页需要登录且返回 403，项目按合规红线不绕验证码、
不携带登录态。图片从官方 dataset.zip 精确抽取。

## 字段
| 字段 | 说明 |
|---|---|
| post_id | 帖文不可逆哈希，跨表关联用 |
| user_id | 发布账号的 sha256 前 16 位，稳定但不可逆；原始账号名默认不落盘 |
| user_id_type | 固定为 sha256_pseudonym |
| time | 发布时间（精确到分钟，2020 年） |
| date | 日期（只到天，供趋势分析） |
| text | 清洗后正文（去 URL/@/微博尾巴） |
| label / label_name | -1 负面 / 0 中性 / 1 正面 |
| image_file | 本地图片相对路径 |
| context | JSON：`{{"text": 清洗后正文, "image": 本地图片相对路径}}` |
| image_sha256 / image_width / image_height / image_bytes | 图片完整性信息 |
| source / dataset_doi | 公开数据来源与 DOI |

## 隐私与合规
- 原始账号名只在内存中生成不可逆 user_id，默认不落盘；
- 输出表不含原始微博 ID、图片 URL、账号名；
- `_internal_provenance.csv` 含原始帖文 ID 与图片 URL，仅供内部溯源，不得提交；
- 如需内部核对账号，运行 `--keep-raw-account` 生成 `_internal_account_map.csv`，同样不得提交。

## 复现
```bash
python src/01c_export_crawl_1000.py --n 1000
```

## 统计
- 条数：{len(df):,}
- 唯一 user_id：{df['user_id'].nunique():,}
- 类别：{df['label'].value_counts().sort_index().to_dict()}
- 日期范围：{df['date'].min()} ~ {df['date'].max()}
- 图片：{len(df):,} 张，合计 {df['image_bytes'].sum() / 1024 / 1024:.1f} MB
"""
    with open(os.path.join(CRAWL_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    manifest = {
        "generated_at": now_str(),
        "n": int(len(df)),
        "unique_user_id": int(df["user_id"].nunique()),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "columns": [str(c) for c in df.columns],
        "context": "JSON(text,image)",
        "images": int(len(df)),
        "image_missing": int(len(picked) - len(df)),
        "image_total_mb": round(float(df["image_bytes"].sum() / 1024 / 1024), 2),
        "source": "ScienceDB Weibo train_dataset",
        "dataset_doi": DATASET_DOI,
        "license": "CC BY 4.0",
        "privacy": "user_id 为账号名 sha256 前 16 位；原始账号/微博ID/图片URL 仅在内部溯源文件",
        "crawl_time": now_str(),
    }
    write_json(os.path.join(CRAWL_DIR, "manifest.json"), manifest)
    print(f"\n[完成] {len(df):,} 条 -> {os.path.relpath(out_csv)}")
    print(f"  唯一 user_id {df['user_id'].nunique():,}；图片 {len(df):,} 张，"
          f"{df['image_bytes'].sum() / 1024 / 1024:.1f} MB")
    print(f"  说明：{os.path.relpath(os.path.join(CRAWL_DIR, 'README.md'))}")
    if len(df) < args.n:
        print(f"  [WARN] 目标 {args.n:,} 条，实际 {len(df):,} 条（图片缺失）")


if __name__ == "__main__":
    main()
