# -*- coding: utf-8 -*-
"""01a 公开数据集一键导入：ScienceDB 微博图文情感数据集。

流程
----
1. 定位/下载官方 zip（DOI: 10.57760/sciencedb.j00133.00366，CC BY 4.0）
2. 解压主表、100k 文本/日期表、10k 测试表
3. 用公开文本表回填日期（只保留到天）
4. 过滤非法标签、空文本、重复文本；按类分层抽样 4,000×3
5. 从 zip 精确抽取抽样图片到 data/raw/images
6. 生成 70/15/15 分层训练/验证/测试集 + 真实无图外部测试集
7. 写逐行溯源清单（内部文件，不进入报告/大屏）

用法
----
    python src/01a_import_scidb.py --zip "D:/data/dataset.zip"
    python src/01a_import_scidb.py --download          # 无 zip 时自动下载
    python src/01a_import_scidb.py --force             # 强制重建
    python src/01a_import_scidb.py --sample-per-class 100  # 快速调试

注意：原始表含“发布人账号”，本脚本读入后立即丢弃，不写入任何下游文件。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import time
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import (DATASET_DOI, DATASET_LICENSE, DATASET_URL, EXTERNAL_PER_CLASS,
                        MIN_TEXT_LEN, RANDOM_SEED, RAW, SAMPLE_PER_CLASS, now_str,
                        parse_day, post_hash, write_json)
from lib_text import clean_text, normalize_for_match, valid_text

ZIP_URL = ("https://china.scidb.cn/download?fileId=40c81662c009189db2776eef4ac6a442"
           "&fileName=dataset.zip")
ZIP_MD5 = "983c0598f04fcd0a45666960b3ccb602"
ZIP_SIZE = 2237043406
MAIN_CSV = "dataset/Weibo/train_dataset/image_text_label.csv"
TRAIN100K_CSV = "dataset/Weibo/train_dataset/nCoV_100k_train_labled.csv"
TEST10K_CSV = "dataset/Weibo/test_dataset/nCov_10k_test.csv"
IMG_PREFIX = "dataset/Weibo/train_dataset/image/"


def log(msg):
    print(msg, flush=True)


def md5sum(path, chunk=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_zip(dest):
    import requests

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    pos = os.path.getsize(dest) if os.path.exists(dest) else 0
    if pos >= ZIP_SIZE:
        log(f"  zip 已存在且大小完整：{dest}")
        return dest
    log(f"  开始下载：{ZIP_URL}")
    log(f"  目标：{dest}（已下载 {pos / 1024 / 1024:.1f} MB / {ZIP_SIZE / 1024 / 1024:.1f} MB）")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchDataImporter/1.0)"}
    if pos:
        headers["Range"] = f"bytes={pos}-"
    with requests.get(ZIP_URL, headers=headers, stream=True, timeout=120) as r:
        if pos and r.status_code != 206:
            log("  服务器不支持断点续传，从头下载")
            pos = 0
            r.close()
            return download_zip(dest)
        r.raise_for_status()
        mode = "ab" if pos else "wb"
        last = time.time()
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pos += len(chunk)
                if time.time() - last > 10:
                    log(f"    {pos / 1024 / 1024:.1f} / {ZIP_SIZE / 1024 / 1024:.1f} MB")
                    last = time.time()
    if os.path.getsize(dest) != ZIP_SIZE:
        raise RuntimeError(f"下载大小不符：{os.path.getsize(dest)} != {ZIP_SIZE}")
    log("  下载完成，校验 MD5 ...")
    got = md5sum(dest)
    if got != ZIP_MD5:
        raise RuntimeError(f"MD5 不符：{got} != {ZIP_MD5}")
    log("  MD5 校验通过")
    return dest


def find_zip(arg_zip=None):
    candidates = []
    if arg_zip:
        candidates.append(arg_zip)
    candidates += [
        os.path.join(RAW, "dataset.zip"),
        os.environ.get("WEIBO_ZIP", ""),
        r"C:\Users\zzz13\AppData\Local\Temp\opencode\weibo_dataset.zip",
    ]
    for p in candidates:
        if p and os.path.exists(p) and os.path.getsize(p) > 1024 * 1024:
            return p
    return None


def read_csv_smart(path, **kw):
    """严格 UTF-8 优先；失败再用 GB18030（允许少量坏字节替换）。"""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kw)
    except Exception:
        return pd.read_csv(path, encoding="gb18030", encoding_errors="replace", **kw)


def extract_csvs(zf, force=False):
    for name, dest_name in [(MAIN_CSV, "image_text_label.csv"),
                            (TRAIN100K_CSV, "nCoV_100k_train_labled.csv"),
                            (TEST10K_CSV, "nCov_10k_test.csv")]:
        dest = os.path.join(RAW, dest_name)
        if os.path.exists(dest) and not force and os.path.getsize(dest) > 1024:
            log(f"  已存在，跳过：{dest_name}")
            continue
        log(f"  解压 {name} -> {dest_name}")
        with zf.open(name) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)


def build_date_map():
    """从 100k 文本表 + 10k 测试表建立 文本->日期 映射（只保留到天）。"""
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
    if not frames:
        return {}
    all_df = pd.concat(frames, ignore_index=True)
    all_df["key"] = all_df["text"].map(normalize_for_match)
    all_df["day"] = all_df["publish_time"].map(parse_day)
    all_df = all_df[(all_df["key"].str.len() > 0) & all_df["day"].notna()]
    all_df = all_df.drop_duplicates("key", keep="first")
    return dict(zip(all_df["key"], all_df["day"]))


def load_main():
    path = os.path.join(RAW, "image_text_label.csv")
    df = read_csv_smart(path, dtype=str)
    df.columns = ["weibo_id", "text", "image_url", "label"][:len(df.columns)]
    df["label_num"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label_num"].isin([-1, 0, 1])].copy()
    df["label"] = df["label_num"].astype(int)
    df["text"] = df["text"].fillna("").astype(str)
    df["image_url"] = df["image_url"].fillna("").astype(str)
    df["key"] = df["text"].map(normalize_for_match)
    df["valid"] = df["text"].map(valid_text)
    df["text_clean"] = df["text"].map(clean_text)
    df["text_len"] = df["text_clean"].str.len()
    df = df[(df["valid"]) & (df["text_len"] >= MIN_TEXT_LEN) & (df["key"].str.len() > 0)]
    before = len(df)
    df = df.drop_duplicates("key", keep="first").copy()
    log(f"  主表有效行 {before:,}，去重后 {len(df):,}（删除重复 {before - len(df):,}）")
    return df


def available_images(zf):
    names = set(zf.namelist())
    # 只保留图片，减少内存
    imgs = {n for n in names if n.startswith(IMG_PREFIX) and n.lower().endswith((".jpg", ".jpeg", ".png"))}
    log(f"  zip 内图片 {len(imgs):,} 张")
    return imgs


def choose_image_name(row, imgs):
    weibo_id = str(row["weibo_id"])
    candidates = [
        f"{IMG_PREFIX}{weibo_id}.jpg",
        f"{IMG_PREFIX}{weibo_id}.jpeg",
        f"{IMG_PREFIX}{weibo_id}.png",
        f"{IMG_PREFIX}{row['row_index']}_{weibo_id}.jpg",
    ]
    for c in candidates:
        if c in imgs:
            return c
    return None


def extract_images(zf, rows, imgs, out_dir, force=False):
    """图片以 post_hash 命名，避免原始微博 ID 进入下游数据。"""
    os.makedirs(out_dir, exist_ok=True)
    done, skip, missing = 0, 0, []
    for _, row in rows.iterrows():
        name = choose_image_name(row, imgs)
        if name is None:
            missing.append(row["weibo_id"])
            continue
        dest = os.path.join(out_dir, str(row["post_hash"]) + os.path.splitext(name)[1].lower())
        if os.path.exists(dest) and os.path.getsize(dest) > 0 and not force:
            skip += 1
            continue
        try:
            with zf.open(name) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            done += 1
        except Exception as e:
            missing.append(row["weibo_id"])
        if (done + skip) % 2000 == 0:
            log(f"    图片抽取 {done + skip:,}/{len(rows):,}（新写 {done:,}，跳过 {skip:,}）")
    log(f"  图片抽取完成：新写 {done:,}，已存在 {skip:,}，缺失 {len(missing):,}")
    return missing


def sample_main(df, imgs, per_class):
    df = df.copy()
    df["row_index"] = df.index.astype(int)
    df["image_name"] = [choose_image_name(r, imgs) for _, r in df.iterrows()]
    df = df[df["image_name"].notna()].copy()
    log(f"  有对应图片文件的有效行 {len(df):,}")
    counts = df["label"].value_counts()
    log(f"  可用类别分布：{ {int(k): int(v) for k, v in counts.sort_index().items()} }")
    parts = []
    for lab in (-1, 0, 1):
        sub = df[df["label"] == lab]
        n = min(per_class, len(sub))
        if n < per_class:
            log(f"  [WARN] 类别 {lab} 仅 {len(sub):,} 条，抽取 {n:,}（少于目标 {per_class:,}）")
        parts.append(sub.sample(n=n, random_state=RANDOM_SEED))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def make_splits(df):
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"],
                                   random_state=RANDOM_SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"],
                                 random_state=RANDOM_SEED)
    train = train.copy(); val = val.copy(); test = test.copy()
    train["split"] = "train"; val["split"] = "val"; test["split"] = "test"
    return pd.concat([train, val, test]).sort_index()


def build_external(main_keys):
    p100 = os.path.join(RAW, "nCoV_100k_train_labled.csv")
    if not os.path.exists(p100):
        log("  [WARN] 无 100k 表，跳过外部测试集")
        return pd.DataFrame(columns=["post_hash", "text_raw", "label", "date", "source"])
    df = read_csv_smart(p100, dtype=str)
    df.columns = ["weibo_id", "publish_time", "account", "text", "images",
                  "videos", "label"][:len(df.columns)]
    df["label_num"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label_num"].isin([-1, 0, 1])].copy()
    df["label"] = df["label_num"].astype(int)
    df["text"] = df["text"].fillna("").astype(str)
    df["images"] = df["images"].fillna("[]").astype(str)

    def n_img(s):
        return len(re.findall(r"https?://[^\s'\"\],]+", s))

    df["n_img"] = df["images"].map(n_img)
    df = df[df["n_img"] == 0].copy()
    df["key"] = df["text"].map(normalize_for_match)
    df["valid"] = df["text"].map(valid_text)
    df["text_clean"] = df["text"].map(clean_text)
    df = df[(df["valid"]) & (df["text_clean"].str.len() >= MIN_TEXT_LEN)]
    df = df[~df["key"].isin(main_keys)]
    df = df.drop_duplicates("key", keep="first").copy()
    df["date"] = df["publish_time"].map(parse_day)
    log(f"  外部真实无图候选 {len(df):,} 条；类别分布 "
        f"{ {int(k): int(v) for k, v in df['label'].value_counts().sort_index().items()} }")
    parts = []
    for lab in (-1, 0, 1):
        sub = df[df["label"] == lab]
        n = min(EXTERNAL_PER_CLASS, len(sub))
        if n < EXTERNAL_PER_CLASS:
            log(f"  [WARN] 外部集类别 {lab} 仅 {len(sub):,} 条")
        parts.append(sub.sample(n=n, random_state=RANDOM_SEED + 1))
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=RANDOM_SEED + 1)
    out["post_hash"] = [post_hash(r["weibo_id"], r["text"]) for _, r in out.iterrows()]
    out["text_raw"] = out["text"]
    out["source"] = "ScienceDB/nCoV_100k_train_labled（无图子集）"
    return out[["post_hash", "text_raw", "label", "date", "source"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="本地 dataset.zip 路径")
    ap.add_argument("--download", action="store_true", help="无 zip 时从 ScienceDB 下载")
    ap.add_argument("--force", action="store_true", help="强制重建")
    ap.add_argument("--sample-per-class", type=int, default=SAMPLE_PER_CLASS)
    args = ap.parse_args()

    print("=" * 66)
    print("01a 公开数据集导入：ScienceDB 微博图文情感数据集")
    print("=" * 66)
    log(f"  DOI：{DATASET_URL}  许可：{DATASET_LICENSE}")

    posts_path = os.path.join(RAW, "posts_raw.csv")
    img_dir = os.path.join(RAW, "images")
    n_existing = len([f for f in os.listdir(img_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]) if os.path.isdir(img_dir) else 0
    if (os.path.exists(posts_path) and n_existing >= 10000 and not args.force
            and args.sample_per_class == SAMPLE_PER_CLASS):
        log(f"  已存在 posts_raw.csv 与 {n_existing:,} 张图片，跳过导入（--force 可重建）")
        return

    zip_path = find_zip(args.zip)
    if zip_path is None:
        if args.download:
            zip_path = download_zip(os.path.join(RAW, "dataset.zip"))
        else:
            log("  [FAIL] 未找到 dataset.zip。请用 --zip 指定，或加 --download 从 ScienceDB 下载。")
            sys.exit(1)
    log(f"  使用 zip：{zip_path}（{os.path.getsize(zip_path) / 1024 / 1024:.1f} MB）")

    with zipfile.ZipFile(zip_path) as zf:
        log("\n[1] 解压 CSV")
        extract_csvs(zf, force=args.force)

        log("\n[2] 读取主表并清洗")
        main_df = load_main()

        log("\n[3] 回填日期（文本精确匹配公开日期表）")
        date_map = build_date_map()
        main_df["date"] = main_df["key"].map(date_map)
        main_df["date_source"] = main_df["date"].map(
            lambda x: "text_match_public_date_table" if pd.notna(x) else "")
        coverage = main_df["date"].notna().mean()
        log(f"  日期覆盖 {main_df['date'].notna().sum():,} / {len(main_df):,}（{coverage:.2%}）")

        log("\n[4] 分层抽样")
        imgs = available_images(zf)
        sampled = sample_main(main_df, imgs, args.sample_per_class)
        sampled["post_hash"] = [post_hash(r["weibo_id"], r["text"]) for _, r in sampled.iterrows()]
        log(f"  抽样 {len(sampled):,} 条："
            f"{ {int(k): int(v) for k, v in sampled['label'].value_counts().sort_index().items()} }")

        log("\n[5] 抽取图片（以 post_hash 命名，不暴露原始 ID）")
        missing = extract_images(zf, sampled, imgs, img_dir, force=args.force)
        if missing:
            log(f"  缺失图片 {len(missing):,} 条，将从抽样集中剔除")
            sampled = sampled[~sampled["weibo_id"].isin(set(missing))].copy()

    log("\n[6] 划分训练/验证/测试集")
    sampled = make_splits(sampled)
    log(f"  train={int((sampled['split'] == 'train').sum()):,}  "
        f"val={int((sampled['split'] == 'val').sum()):,}  "
        f"test={int((sampled['split'] == 'test').sum()):,}")

    sampled["source"] = "ScienceDB/Weibo train_dataset（公开图文子集）"
    sampled["image_file"] = sampled["post_hash"].astype(str) + ".jpg"
    posts = sampled[["post_hash", "row_index", "text", "image_file", "label",
                     "date", "date_source", "split", "source"]].rename(columns={"text": "text_raw"})
    posts.to_csv(posts_path, index=False, encoding="utf-8-sig")

    # 逐行溯源（内部文件：含原始 ID 与图片 URL，不进入报告/大屏）
    prov = sampled[["post_hash", "row_index", "weibo_id", "image_url", "image_file", "label",
                    "date", "date_source", "split"]].copy()
    prov["dataset_doi"] = DATASET_DOI
    prov.to_csv(os.path.join(RAW, "provenance.csv"), index=False, encoding="utf-8-sig")

    log("\n[7] 构建真实无图外部测试集")
    ext = build_external(set(main_df["key"]))
    ext.to_csv(os.path.join(RAW, "external_text_only.csv"), index=False, encoding="utf-8-sig")
    log(f"  外部测试集 {len(ext):,} 条")

    manifest = {
        "imported_at": now_str(),
        "dataset_title": "Weibo and Twitter datasets（微博图文子集）",
        "dataset_doi": DATASET_DOI,
        "dataset_url": DATASET_URL,
        "license": DATASET_LICENSE,
        "zip_path": zip_path,
        "zip_size": os.path.getsize(zip_path),
        "zip_md5": ZIP_MD5,
        "raw_main_rows": int(len(pd.read_csv(os.path.join(RAW, "image_text_label.csv"),
                                             dtype=str, usecols=[0]).dropna())),
        "valid_rows_after_clean": int(len(main_df)),
        "date_coverage": round(float(coverage), 6),
        "dated_rows": int(main_df["date"].notna().sum()),
        "sample_rows": int(len(posts)),
        "sample_label_counts": {str(k): int(v) for k, v in posts["label"].value_counts().sort_index().items()},
        "split_counts": {str(k): int(v) for k, v in posts["split"].value_counts().items()},
        "image_files": len([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]),
        "external_rows": int(len(ext)),
        "external_label_counts": {str(k): int(v) for k, v in ext["label"].value_counts().sort_index().items()},
        "random_seed": RANDOM_SEED,
        "privacy": "原始账号列读取后立即丢弃；下游仅保留 post_hash；日期只到天；图片 URL 仅存内部溯源文件",
    }
    write_json(os.path.join(RAW, "import_manifest.json"), manifest)
    log(f"\n[完成] 抽样 {len(posts):,} 条，图片 {manifest['image_files']:,} 张，"
        f"日期覆盖 {coverage:.2%}，外部无图 {len(ext):,} 条")
    log(f"  清单：data/raw/import_manifest.json")


if __name__ == "__main__":
    main()
