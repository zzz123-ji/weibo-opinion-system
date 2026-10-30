# -*- coding: utf-8 -*-
"""02 预处理：清洗、匿名化、日期回填、图像特征提取、数据集划分。

输入
----
data/raw/image_text_label.csv      64,347 条公开图文帖（含账号列，读取后丢弃）
data/raw/nCoV_100k_train_labled.csv 公开文本/日期表（仅用于回填日期）
data/raw/nCov_10k_test.csv          公开文本/日期表
data/raw/posts_raw.csv              01a 分层抽样 12,000 条
data/raw/images/*.jpg               以 post_hash 命名的抽样图片

输出
----
data/processed/posts_full.csv       清洗后的全量 5.6 万条（趋势/话题分析用）
data/processed/posts_clean.csv      建模用 12,000 条（含 split、image_row）
data/processed/external_test_clean.csv 真实无图外部测试 3,000 条
data/processed/image_features.npy   ResNet18 512 维图像特征
data/processed/image_feature_index.csv
data/processed/image_stats.csv      图像质量统计（亮度/饱和度/边缘等）
output/preprocess_report.json
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import (LABEL_NAMES, MIN_TEXT_LEN, PROCESSED, RAW, RANDOM_SEED,
                        TABLES, now_str, parse_day, post_hash, write_json)
from lib_image import ImageEncoder, image_stats, load_image
from lib_text import (clean_text, extract_hashtags, extract_provinces,
                      normalize_for_match, text_stats, valid_text)


def read_csv_smart(path, **kw):
    """严格 UTF-8 优先；失败再用 GB18030（允许少量坏字节替换）。"""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kw)
    except Exception:
        return pd.read_csv(path, encoding="gb18030", encoding_errors="replace", **kw)


def build_date_map():
    frames = []
    p100 = os.path.join(RAW, "nCoV_100k_train_labled.csv")
    if os.path.exists(p100):
        df = read_csv_smart(p100, dtype=str)
        df.columns = ["weibo_id", "publish_time", "account", "text", "images",
                      "videos", "label"][:len(df.columns)]
        frames.append(df[["text", "publish_time"]])
    p10 = os.path.join(RAW, "nCov_10k_test.csv")
    if os.path.exists(p10):
        df = read_csv_smart(p10, dtype=str)
        df.columns = ["weibo_id", "publish_time", "account", "text", "images",
                      "videos"][:len(df.columns)]
        frames.append(df[["text", "publish_time"]])
    all_df = pd.concat(frames, ignore_index=True)
    all_df["key"] = all_df["text"].map(normalize_for_match)
    all_df["day"] = all_df["publish_time"].map(parse_day)
    all_df = all_df[(all_df["key"].str.len() > 0) & all_df["day"].notna()]
    all_df = all_df.drop_duplicates("key", keep="first")
    return dict(zip(all_df["key"], all_df["day"]))


def enrich(df):
    df = df.copy()
    df["text_clean"] = df["text_raw"].map(clean_text)
    df["text_len"] = df["text_clean"].str.len()
    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map(LABEL_NAMES)
    df["has_hashtag"] = df["text_raw"].map(lambda x: bool(extract_hashtags(x)))
    df["has_mention"] = df["text_raw"].astype(str).str.contains("@", regex=False)
    df["n_hashtags"] = df["text_raw"].map(lambda x: len(extract_hashtags(x)))
    df["hashtags"] = df["text_raw"].map(lambda x: "|".join(extract_hashtags(x)[:8]))
    df["provinces"] = df["text_raw"].map(lambda x: "|".join(extract_provinces(x)[:8]))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-backend", default="resnet18",
                    choices=["resnet18", "handcrafted"])
    ap.add_argument("--skip-images", action="store_true", help="仅调试，不提取图像特征")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 66)
    print("02 预处理：清洗 / 匿名化 / 图像特征 / 数据集划分")
    print("=" * 66)

    print("\n[1] 全量主表清洗")
    raw = read_csv_smart(os.path.join(RAW, "image_text_label.csv"), dtype=str)
    raw.columns = ["weibo_id", "text_raw", "image_url", "label"][:len(raw.columns)]
    raw["label_num"] = pd.to_numeric(raw["label"], errors="coerce")
    raw = raw[raw["label_num"].isin([-1, 0, 1])].copy()
    raw["label"] = raw["label_num"].astype(int)
    raw["label_name"] = raw["label"].map(LABEL_NAMES)
    raw["text_raw"] = raw["text_raw"].fillna("").astype(str)
    raw["text_clean"] = raw["text_raw"].map(clean_text)
    raw["text_len"] = raw["text_clean"].str.len()
    raw["key"] = raw["text_raw"].map(normalize_for_match)
    raw["valid"] = raw["text_raw"].map(valid_text)
    raw = raw[raw["valid"] & (raw["key"].str.len() > 0)
              & (raw["text_len"] >= MIN_TEXT_LEN)].copy()
    n_valid = len(raw)
    raw = raw.drop_duplicates("key", keep="first").copy()
    raw["row_index"] = raw.index.astype(int)
    print(f"  合法标签/正文 {n_valid:,} 条，文本去重后 {len(raw):,} 条")

    print("\n[2] 回填日期（只到天）")
    dmap = build_date_map()
    raw["date"] = raw["key"].map(dmap)
    coverage = raw["date"].notna().mean()
    print(f"  日期覆盖 {raw['date'].notna().sum():,} / {len(raw):,}（{coverage:.2%}）")

    print("\n[3] 抽样表关联与清洗")
    posts = pd.read_csv(os.path.join(RAW, "posts_raw.csv"), dtype=str)
    posts["label"] = posts["label"].astype(int)
    posts = enrich(posts)
    raw["post_hash"] = [post_hash(r["weibo_id"], r["text_raw"]) for _, r in raw.iterrows()]
    sample_hashes = set(posts["post_hash"])
    raw["in_sample"] = raw["post_hash"].isin(sample_hashes)
    print(f"  全量表中命中抽样 {int(raw['in_sample'].sum()):,} / {len(posts):,}")
    if int(raw["in_sample"].sum()) != len(posts):
        raise RuntimeError("抽样表与全量表 post_hash 不一致，请重跑 01a")

    # 全量表：趋势/话题分析
    full = raw[["post_hash", "row_index", "text_clean", "text_len", "label", "label_name",
                "date", "in_sample"]].copy()
    full["has_hashtag"] = raw["text_raw"].map(lambda x: bool(extract_hashtags(x))).values
    full["has_mention"] = raw["text_raw"].astype(str).str.contains("@", regex=False).values
    full["n_hashtags"] = raw["text_raw"].map(lambda x: len(extract_hashtags(x))).values
    full["hashtags"] = raw["text_raw"].map(lambda x: "|".join(extract_hashtags(x)[:8])).values
    full["provinces"] = raw["text_raw"].map(lambda x: "|".join(extract_provinces(x)[:8])).values
    full["has_image_url"] = True
    full.to_csv(os.path.join(PROCESSED, "posts_full.csv"), index=False, encoding="utf-8-sig")
    print(f"  全量表输出 {len(full):,} 条 -> data/processed/posts_full.csv")

    # 抽样表补日期（若 01a 未回填到）
    posts["date"] = posts["date"].fillna(posts["post_hash"].map(
        dict(zip(raw["post_hash"], raw["date"]))))
    posts["label_name"] = posts["label"].map(LABEL_NAMES)
    posts["image_path"] = posts["image_file"].map(lambda x: os.path.join(RAW, "images", x))
    missing_files = int((~posts["image_path"].map(os.path.exists)).sum())
    print(f"  抽样图片缺失 {missing_files} 张")

    print("\n[4] 图像特征提取")
    feature_path = os.path.join(PROCESSED, "image_features.npy")
    stats_path = os.path.join(PROCESSED, "image_stats.csv")
    bad_idx = []
    backend = args.image_backend
    dim = 512 if backend == "resnet18" else 96
    if args.skip_images and os.path.exists(feature_path):
        print("  --skip-images：复用已有图像特征")
        dim = int(np.load(feature_path, mmap_mode="r").shape[1])
    else:
        enc = ImageEncoder(backend)
        backend = enc.backend
        dim = enc.dim
        t1 = time.time()
        X, bad_idx = enc.encode_paths(posts["image_path"].tolist(), batch_size=64)
        np.save(feature_path, X.astype(np.float32))
        print(f"  图像特征 {X.shape}，backend={backend}，耗时 {time.time() - t1:.1f}s，"
              f"损坏 {len(bad_idx)} 张")

    # 图像质量统计（可解释分析用）
    stat_rows = []
    for _, r in posts.iterrows():
        im = load_image(r["image_path"])
        s = image_stats(im)
        s.update({"post_hash": r["post_hash"], "label": int(r["label"]),
                  "split": r["split"]})
        stat_rows.append(s)
    pd.DataFrame(stat_rows).to_csv(stats_path, index=False, encoding="utf-8-sig")
    print(f"  图像质量统计 {len(stat_rows):,} 条 -> data/processed/image_stats.csv")

    posts = posts.reset_index(drop=True)
    posts["image_row"] = posts.index.astype(int)
    feature_index = posts[["post_hash", "image_row", "image_file", "label", "split"]].copy()
    feature_index["backend"] = backend
    feature_index["dim"] = dim
    feature_index.to_csv(os.path.join(PROCESSED, "image_feature_index.csv"),
                         index=False, encoding="utf-8-sig")

    keep = ["post_hash", "row_index", "text_clean", "text_len", "label", "label_name",
            "date", "split", "image_file", "image_row", "has_hashtag", "has_mention",
            "n_hashtags", "hashtags", "provinces", "source"]
    posts[keep].to_csv(os.path.join(PROCESSED, "posts_clean.csv"),
                       index=False, encoding="utf-8-sig")
    print(f"  建模表 {len(posts):,} 条 -> data/processed/posts_clean.csv")

    print("\n[5] 外部真实无图测试集清洗")
    ext = pd.read_csv(os.path.join(RAW, "external_text_only.csv"), dtype=str)
    ext = enrich(ext)
    ext["label_name"] = ext["label"].astype(int).map(LABEL_NAMES)
    ext["date"] = ext["date"].map(parse_day)
    ext_keep = ["post_hash", "text_clean", "text_len", "label", "label_name", "date",
                "has_hashtag", "has_mention", "n_hashtags", "hashtags", "provinces", "source"]
    ext[ext_keep].to_csv(os.path.join(PROCESSED, "external_test_clean.csv"),
                         index=False, encoding="utf-8-sig")
    print(f"  外部测试 {len(ext):,} 条 -> data/processed/external_test_clean.csv")

    print("\n[6] 预处理报告")
    report = {
        "generated_at": now_str(),
        "raw_rows": int(len(pd.read_csv(os.path.join(RAW, "image_text_label.csv"),
                                        dtype=str, usecols=[0]))),
        "valid_label_text_rows": int(n_valid),
        "dedup_rows": int(len(raw)),
        "duplicates_removed": int(n_valid - len(raw)),
        "date_coverage": round(float(coverage), 6),
        "dated_rows": int(raw["date"].notna().sum()),
        "date_min": str(raw["date"].dropna().min()) if raw["date"].notna().any() else "",
        "date_max": str(raw["date"].dropna().max()) if raw["date"].notna().any() else "",
        "sample_rows": int(len(posts)),
        "sample_label_counts": {str(k): int(v) for k, v in posts["label"].value_counts().sort_index().items()},
        "split_counts": {str(k): int(v) for k, v in posts["split"].value_counts().items()},
        "sample_date_coverage": round(float(posts["date"].notna().mean()), 6),
        "missing_images": missing_files,
        "bad_images": len(bad_idx),
        "image_backend": backend,
        "image_dim": int(dim),
        "text_len_mean": round(float(posts["text_len"].mean()), 2),
        "text_len_median": int(posts["text_len"].median()),
        "text_len_p95": int(posts["text_len"].quantile(0.95)),
        "external_rows": int(len(ext)),
        "external_label_counts": {str(k): int(v) for k, v in ext["label"].value_counts().sort_index().items()},
        "privacy": "不含账号/昵称/原始微博ID；日期只到天；图片以 post_hash 命名",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    write_json(os.path.join(os.path.dirname(TABLES), "preprocess_report.json"), report)
    pd.DataFrame([{"指标": k, "值": v} for k, v in report.items()
                  if not isinstance(v, dict)]).to_csv(
        os.path.join(TABLES, "预处理统计.csv"), index=False, encoding="utf-8-sig")
    print(f"  报告：output/preprocess_report.json，总耗时 {report['elapsed_seconds']}s")
    print("\n[完成] 预处理完成。")


if __name__ == "__main__":
    main()
