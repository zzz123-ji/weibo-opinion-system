# -*- coding: utf-8 -*-
"""01 数据校验：按数据字典检查公开数据集导入结果，输出质量报告。

用法
----
    python src/01_data_collection.py --check
    python src/01_data_collection.py --source   # 打印数据来源与合规说明

本脚本不生成任何仿真数据；data/raw 必须来自 01a_import_scidb.py 的公开数据集导入。
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import (DATASET_DOI, DATASET_LICENSE, DATASET_TITLE, DATASET_URL,
                        LABEL_ORDER, RAW, TABLES, now_str, read_json, write_json)

RAW_MAIN = os.path.join(RAW, "image_text_label.csv")
RAW_100K = os.path.join(RAW, "nCoV_100k_train_labled.csv")
RAW_10K = os.path.join(RAW, "nCov_10k_test.csv")
RAW_POSTS = os.path.join(RAW, "posts_raw.csv")
RAW_EXT = os.path.join(RAW, "external_text_only.csv")
IMG_DIR = os.path.join(RAW, "images")

SCHEMA = {
    "image_text_label.csv": ["微博id", "微博中文内容", "微博图片", "情感倾向"],
    "nCoV_100k_train_labled.csv": ["微博id", "微博发布时间", "发布人账号",
                                    "微博中文内容", "微博图片", "微博视频", "情感倾向"],
    "nCov_10k_test.csv": ["微博id", "微博发布时间", "发布人账号",
                           "微博中文内容", "微博图片", "微博视频"],
}
POSTS_COLUMNS = ["post_hash", "row_index", "text_raw", "image_file", "label",
                 "date", "date_source", "split", "source"]
EXT_COLUMNS = ["post_hash", "text_raw", "label", "date", "source"]

FAILS, WARNS = [], []


def ok(msg):
    print(f"  [PASS] {msg}")


def fail(msg):
    FAILS.append(msg)
    print(f"  [FAIL] {msg}")


def warn(msg):
    WARNS.append(msg)
    print(f"  [WARN] {msg}")


def info(msg):
    print(f"  [INFO] {msg}")


def check_file(path, columns, label):
    if not os.path.exists(path):
        fail(f"{label} 不存在：{os.path.relpath(path)}（先运行 src/01a_import_scidb.py）")
        return None
    df = None
    last_err = None
    for enc in ("utf-8-sig", "gb18030"):
        try:
            df = pd.read_csv(path, dtype=str, nrows=5, encoding=enc)
            break
        except Exception as e:
            last_err = e
            if enc == "gb18030":
                try:
                    df = pd.read_csv(path, dtype=str, nrows=5, encoding=enc,
                                     encoding_errors="replace")
                    break
                except Exception as e2:
                    last_err = e2
    if df is None:
        fail(f"{label} 无法读取：{last_err}")
        return None
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(f"{label} 缺列：{missing}")
    else:
        ok(f"{label} 列名符合数据字典（{len(columns)} 列）")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="执行校验（默认）")
    parser.add_argument("--source", action="store_true", help="打印数据来源")
    args = parser.parse_args()

    print("=" * 66)
    print("01 数据校验：微博图文情感公开数据集")
    print("=" * 66)

    if args.source:
        print(f"数据集：{DATASET_TITLE}")
        print(f"DOI   ：{DATASET_URL}")
        print(f"许可  ：{DATASET_LICENSE}")
        print("获取  ：ScienceDB 公开文件 API + 官方下载直链，不爬取微博站内接口")
        print("隐私  ：原始表含发布人账号，导入时立即丢弃；正文去 @/URL；日期只到天")
        return

    quality = {"checked_at": now_str(), "dataset_doi": DATASET_DOI,
               "dataset_license": DATASET_LICENSE, "checks": []}

    def record(name, status, detail):
        quality["checks"].append({"check": name, "status": status, "detail": detail})

    print("\n[1] 原始数据表")
    main_df = check_file(RAW_MAIN, SCHEMA["image_text_label.csv"], "图文主表")
    check_file(RAW_100K, SCHEMA["nCoV_100k_train_labled.csv"], "100k 文本/日期表")
    check_file(RAW_10K, SCHEMA["nCov_10k_test.csv"], "10k 测试表")

    if main_df is not None:
        df = pd.read_csv(RAW_MAIN, dtype=str, keep_default_na=False)
        n = len(df)
        quality["raw_main_rows"] = n
        if n >= 60000:
            ok(f"图文主表行数 {n:,}（≥60,000）")
            record("图文主表行数", "PASS", n)
        else:
            fail(f"图文主表行数仅 {n:,}（<60,000）")
            record("图文主表行数", "FAIL", n)

        labels = pd.to_numeric(df["情感倾向"], errors="coerce")
        invalid_mask = ~labels.isin([-1, 0, 1])
        invalid = int(invalid_mask.sum())
        counts = {str(k): int(v) for k, v in labels[labels.isin([-1, 0, 1])]
                  .astype(int).value_counts().sort_index().items()}
        quality["raw_label_counts"] = counts
        quality["raw_invalid_labels"] = invalid
        if invalid / max(n, 1) <= 0.001:
            ok(f"标签取值合法（-1/0/1），异常 {invalid} 条（{invalid / n:.3%}，导入时剔除）")
            record("标签取值", "PASS", f"异常 {invalid} 条")
        else:
            fail(f"标签异常 {invalid} 条（>0.1%）")
            record("标签取值", "FAIL", f"异常 {invalid} 条")

        text_nonempty = int((df["微博中文内容"].astype(str).str.strip().str.len() > 0).sum())
        img_nonempty = int((df["微博图片"].astype(str).str.strip().str.len() > 0).sum())
        quality["raw_text_nonempty"] = text_nonempty
        quality["raw_image_nonempty"] = img_nonempty
        if text_nonempty / max(n, 1) >= 0.99:
            ok(f"正文非空率 {text_nonempty / n:.2%}")
            record("正文非空率", "PASS", f"{text_nonempty / n:.4f}")
        else:
            warn(f"正文非空率仅 {text_nonempty / n:.2%}")
            record("正文非空率", "WARN", f"{text_nonempty / n:.4f}")
        if img_nonempty / max(n, 1) >= 0.99:
            ok(f"图片 URL 非空率 {img_nonempty / n:.2%}")
            record("图片非空率", "PASS", f"{img_nonempty / n:.4f}")
        else:
            warn(f"图片 URL 非空率仅 {img_nonempty / n:.2%}")
            record("图片非空率", "WARN", f"{img_nonempty / n:.4f}")

        dup = int(df["微博id"].duplicated().sum())
        quality["raw_dup_ids"] = dup
        if dup == 0:
            ok("微博 ID 无重复")
            record("ID 唯一性", "PASS", 0)
        else:
            warn(f"微博 ID 重复 {dup} 条")
            record("ID 唯一性", "WARN", dup)

    print("\n[2] 抽样与图片")
    posts_df = check_file(RAW_POSTS, POSTS_COLUMNS, "建模抽样表")
    if posts_df is not None:
        p = pd.read_csv(RAW_POSTS, dtype=str, keep_default_na=False)
        quality["sample_rows"] = len(p)
        quality["sample_label_counts"] = {str(k): int(v) for k, v in
                                          p["label"].value_counts().sort_index().items()}
        quality["sample_split_counts"] = {str(k): int(v) for k, v in
                                          p["split"].value_counts().items()}
        if len(p) == 12000:
            ok(f"抽样表 {len(p):,} 条（4,000/类 × 3 类）")
            record("抽样规模", "PASS", len(p))
        else:
            fail(f"抽样表 {len(p):,} 条（期望 12,000）")
            record("抽样规模", "FAIL", len(p))
        bad_cols = [c for c in p.columns if c in {"微博id", "发布人账号", "account", "user_id", "nickname"}]
        if bad_cols:
            fail(f"抽样表含敏感/原始账号列：{bad_cols}")
            record("隐私列", "FAIL", str(bad_cols))
        else:
            ok("抽样表不含账号/昵称/原始 ID 列")
            record("隐私列", "PASS", "无")

        n_img = len([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]) \
            if os.path.isdir(IMG_DIR) else 0
        quality["image_files"] = n_img
        if n_img >= len(p) * 0.98:
            ok(f"本地图片 {n_img:,} 张（抽样 {len(p):,} 条）")
            record("图片文件", "PASS", n_img)
        else:
            fail(f"本地图片仅 {n_img:,} 张（抽样 {len(p):,} 条）")
            record("图片文件", "FAIL", n_img)

        if n_img:
            random.seed(42)
            files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            sample = random.sample(files, min(20, len(files)))
            bad = 0
            for f in sample:
                try:
                    with Image.open(os.path.join(IMG_DIR, f)) as im:
                        im.verify()
                except Exception:
                    bad += 1
            if bad == 0:
                ok("随机 20 张图片可正常解码")
                record("图片可读性", "PASS", "20/20")
            else:
                fail(f"随机 20 张图片中 {bad} 张损坏")
                record("图片可读性", "FAIL", f"{bad}/20")

    print("\n[3] 外部真实无图测试集")
    ext_df = check_file(RAW_EXT, EXT_COLUMNS, "外部无图测试表")
    if ext_df is not None:
        e = pd.read_csv(RAW_EXT, dtype=str, keep_default_na=False)
        quality["external_rows"] = len(e)
        quality["external_label_counts"] = {str(k): int(v) for k, v in
                                            e["label"].value_counts().sort_index().items()}
        if len(e) >= 3000:
            ok(f"外部测试集 {len(e):,} 条（真实无图帖文）")
            record("外部测试集", "PASS", len(e))
        else:
            warn(f"外部测试集仅 {len(e):,} 条（<3,000）")
            record("外部测试集", "WARN", len(e))
        if "image_file" in e.columns or "微博id" in e.columns:
            fail("外部测试集不应含图片或原始 ID 字段")
            record("外部测试隐私", "FAIL", "含图片/ID")
        else:
            ok("外部测试集仅含文本、标签、日期")
            record("外部测试隐私", "PASS", "无图片/ID")

    print("\n[4] 结果汇总")
    quality["fails"] = FAILS
    quality["warns"] = WARNS
    quality["status"] = "PASS" if not FAILS else "FAIL"
    write_json(os.path.join(os.path.dirname(TABLES), "data_quality.json"), quality)

    rows = [{"检查项": c["check"], "结果": c["status"], "详情": str(c["detail"])}
            for c in quality["checks"]]
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, "数据校验报告.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"\n  PASS {len(quality['checks']) - len(FAILS) - len(WARNS)}  "
          f"WARN {len(WARNS)}  FAIL {len(FAILS)}")
    print(f"  质量报告：output/data_quality.json")
    if FAILS:
        print("\n[阻断] 数据校验未通过，请先修复。")
        sys.exit(1)
    print("\n[通过] 数据校验完成。")


if __name__ == "__main__":
    main()
