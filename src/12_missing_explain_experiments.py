# -*- coding: utf-8 -*-
"""12 缺失比例/位置曲线 + 解释删除实验。

对应作品方案 E2 与 E4 的可运行子集：
- E2 缺失比例：按样本随机屏蔽 10%/30%/50%/70% 的文本、图像或双模态；
- E2 缺失位置：当前特征是静态向量，用“特征维连续区间置零”模拟开头/中间/结尾
  的局部缺失，明确标注为代理实验，不等同于视频时间轴位置；
- E4 删除实验：文本按 token 重要性删除 vs 随机删除；图像按 3×3 区域重要性
  删除 vs 随机删除，比较真实类概率下降。

输出
----
output/missing_experiments.json
output/tables/缺失比例曲线.csv
output/tables/缺失位置曲线.csv
output/tables/删除实验.csv
docs/figures/fig11_missing_ratio.png
docs/figures/fig12_deletion.png

并把 missing_ratio / missing_position / deletion 合并进
output/models_metrics.json 与 output/explainability.json。

边界
----
- 缺失比例是“样本级整模态缺失”，不是视频帧级缺失；
- 缺失位置是“特征维连续缺失”的代理，不是真实时间片段；
- 删除实验是删除式敏感性分析，不是因果证明；
- 所有实验只使用固定 test split，不参与训练或调参。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import ImageDraw
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import (DOCS, MODELS, OUT, PROCESSED, ROOT, TABLES, now_str,
                        read_json, write_json)
from lib_image import image_stats, load_image
from lib_infer import SentimentInference
from lib_text import tokenize_keywords

SEED = 42
CLASSES = np.array([-1, 0, 1])
RATIOS = [0.1, 0.3, 0.5, 0.7]
SEEDS = [SEED, SEED + 1, SEED + 2]
POSITIONS = ["beginning", "middle", "end"]
POS_NAMES = {"beginning": "开头", "middle": "中间", "end": "结尾"}


class EarlyFusionMLP(nn.Module):
    """与 04_modeling.py 一致的无门控对照网络（只用于加载权重）。"""

    def __init__(self, d_text=256, d_image=519, hidden=256, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_text + d_image + 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 3),
        )

    def forward(self, zt, zi, mt, mi, force_equal=False):
        mt = mt.view(-1, 1)
        mi = mi.view(-1, 1)
        zt = zt * mt
        zi = zi * mi
        x = torch.cat([zt, zi, mt, mi], dim=1)
        return self.net(x), torch.cat([mt, mi], dim=1)


def predict_gated(eng: SentimentInference, Z: np.ndarray, I: np.ndarray,
                  mt: np.ndarray, mi: np.ndarray, batch_size: int = 512) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(Z), batch_size):
            zt = torch.from_numpy(Z[start:start + batch_size].astype(np.float32))
            zi = torch.from_numpy(I[start:start + batch_size].astype(np.float32))
            m1 = torch.from_numpy(mt[start:start + batch_size].astype(np.float32))
            m2 = torch.from_numpy(mi[start:start + batch_size].astype(np.float32))
            logits = eng.model(zt, zi, m1, m2)[0]
            out.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(out)


def predict_early(model: nn.Module, Z: np.ndarray, I: np.ndarray,
                  mt: np.ndarray, mi: np.ndarray, batch_size: int = 512) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(Z), batch_size):
            zt = torch.from_numpy(Z[start:start + batch_size].astype(np.float32))
            zi = torch.from_numpy(I[start:start + batch_size].astype(np.float32))
            m1 = torch.from_numpy(mt[start:start + batch_size].astype(np.float32))
            m2 = torch.from_numpy(mi[start:start + batch_size].astype(np.float32))
            logits = model(zt, zi, m1, m2)[0]
            out.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(out)


def metric(y: np.ndarray, p: np.ndarray) -> dict:
    pred = CLASSES[p.argmax(1)]
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 6),
        "macro_f1": round(float(f1_score(y, pred, average="macro",
                                         labels=CLASSES, zero_division=0)), 6),
    }


def mask_block(X: np.ndarray, ratio: float, position: str) -> np.ndarray:
    """把特征维的一段连续区间置零，模拟局部缺失（静态特征代理）。"""
    X2 = X.copy()
    d = X2.shape[1]
    k = max(1, int(round(d * ratio)))
    if position == "beginning":
        start = 0
    elif position == "middle":
        start = max(0, (d - k) // 2)
    else:
        start = max(0, d - k)
    X2[:, start:start + k] = 0.0
    return X2


def load_context():
    eng = SentimentInference()
    df = pd.read_csv(os.path.join(PROCESSED, "posts_clean.csv"), dtype=str)
    df["label"] = df["label"].astype(int)
    df["text_clean"] = df["text_clean"].fillna("")
    df["image_row"] = df["image_row"].astype(int)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    y = test_df["label"].to_numpy()

    Z = eng._text_features(test_df["text_clean"].tolist())
    resnet = np.load(os.path.join(PROCESSED, "image_features.npy")).astype(np.float32)
    resnet = resnet / np.clip(np.linalg.norm(resnet, axis=1, keepdims=True), 1e-6, None)
    stats_df = pd.read_csv(os.path.join(PROCESSED, "image_stats.csv")).set_index("post_hash")
    stats_aligned = stats_df.loc[df["post_hash"], eng.stat_cols].to_numpy(np.float32)
    stats_scaled = (eng.stats_scaler.transform(stats_aligned) * 0.5).astype(np.float32)
    image_all = np.concatenate([resnet, stats_scaled], axis=1).astype(np.float32)
    I = image_all[test_df["image_row"].to_numpy()]

    ckpt = torch.load(os.path.join(MODELS, "early_fusion_mlp.pt"),
                      map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    early = EarlyFusionMLP(
        d_text=int(cfg.get("d_text", 256)),
        d_image=int(cfg.get("d_image", 519)),
        hidden=int(cfg.get("hidden", 256)),
        dropout=float(cfg.get("dropout", 0.25)),
    )
    early.load_state_dict(ckpt["state_dict"])
    early.eval()
    return eng, df, test_df, Z, I, y, early, image_all, resnet, stats_scaled


def missing_ratio_experiment(eng, early, Z, I, y):
    rows = []
    base = {
        "gated": metric(y, predict_gated(eng, Z, I, np.ones(len(y)), np.ones(len(y)))),
        "early_mlp": metric(y, predict_early(early, Z, I, np.ones(len(y)), np.ones(len(y)))),
    }
    for model_name in ["gated", "early_mlp"]:
        for modality in ["text", "image", "both"]:
            for ratio in RATIOS:
                vals = []
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    drop = rng.random(len(y)) < ratio
                    mt = np.ones(len(y), np.float32)
                    mi = np.ones(len(y), np.float32)
                    if modality in ("text", "both"):
                        mt[drop] = 0.0
                    if modality in ("image", "both"):
                        mi[drop] = 0.0
                    if model_name == "gated":
                        p = predict_gated(eng, Z, I, mt, mi)
                    else:
                        p = predict_early(early, Z, I, mt, mi)
                    vals.append(metric(y, p))
                rows.append({
                    "model": model_name,
                    "modality": modality,
                    "ratio": ratio,
                    "accuracy_mean": round(float(np.mean([v["accuracy"] for v in vals])), 6),
                    "accuracy_std": round(float(np.std([v["accuracy"] for v in vals])), 6),
                    "macro_f1_mean": round(float(np.mean([v["macro_f1"] for v in vals])), 6),
                    "macro_f1_std": round(float(np.std([v["macro_f1"] for v in vals])), 6),
                    "n": int(len(y)),
                })
                print(f"  {model_name:<9} {modality:<5} ratio={ratio:.1f} "
                      f"macro-F1={rows[-1]['macro_f1_mean']:.4f}±{rows[-1]['macro_f1_std']:.4f}")
    return {
        "definition": "样本级整模态缺失：按比例随机选择测试样本，将该模态可用性掩码置 0；双模态使用同一批样本。",
        "ratios": RATIOS,
        "seeds": SEEDS,
        "complete": base,
        "rows": rows,
    }


def missing_position_experiment(eng, early, Z, I, y):
    rows = []
    for model_name in ["gated", "early_mlp"]:
        for modality in ["text", "image"]:
            for position in POSITIONS:
                for ratio in RATIOS:
                    if modality == "text":
                        Z2, I2 = mask_block(Z, ratio, position), I
                    else:
                        Z2, I2 = Z, mask_block(I, ratio, position)
                    mt = np.ones(len(y), np.float32)
                    mi = np.ones(len(y), np.float32)
                    p = (predict_gated(eng, Z2, I2, mt, mi) if model_name == "gated"
                         else predict_early(early, Z2, I2, mt, mi))
                    m = metric(y, p)
                    rows.append({
                        "model": model_name, "modality": modality,
                        "position": position, "position_name": POS_NAMES[position],
                        "ratio": ratio, **m,
                    })
                    print(f"  {model_name:<9} {modality:<5} {POS_NAMES[position]} "
                          f"ratio={ratio:.1f} macro-F1={m['macro_f1']:.4f}")
    return {
        "definition": "静态特征代理：将文本 256 维或图像 519 维特征的一段连续区间置零，"
                      "分别从开头/中间/结尾开始；模态掩码保持可用，不是视频时间轴缺失。",
        "positions": POSITIONS,
        "position_names": POS_NAMES,
        "ratios": RATIOS,
        "rows": rows,
    }


def _candidate_tokens(text: str) -> list[str]:
    toks = []
    for w in tokenize_keywords(text):
        if len(w) >= 2 and w in text and w not in toks:
            toks.append(w)
    if len(toks) < 3:
        for span in SentimentInference._fallback_spans(text):
            if span not in toks and span in text:
                toks.append(span)
            if len(toks) >= 8:
                break
    return toks[:8]


def _remove_tokens(text: str, toks: list[str]) -> str:
    out = text
    for t in toks:
        out = out.replace(t, "")
    return out


def _select_correct(eng, Z, I, y, n_total: int):
    p = predict_gated(eng, Z, I, np.ones(len(y)), np.ones(len(y)))
    pred = CLASSES[p.argmax(1)]
    chosen = []
    per_class = max(1, n_total // 3)
    rng = np.random.default_rng(SEED)
    for lab in CLASSES:
        cand = np.where((y == lab) & (pred == lab))[0]
        if len(cand):
            chosen.extend(rng.choice(cand, size=min(per_class, len(cand)),
                                     replace=False).tolist())
    return sorted(set(chosen)), p


def text_deletion_experiment(eng, df, test_df, Z, I, y, n_total: int):
    chosen, base_p = _select_correct(eng, Z, I, y, n_total)
    base_idx = base_p.argmax(1)
    samples = []
    for si in chosen:
        txt = str(test_df.iloc[si]["text_clean"])
        toks = _candidate_tokens(txt)
        if not toks:
            continue
        pert_texts, used = [], []
        for tok in toks:
            new = _remove_tokens(txt, [tok])
            if len(new.strip()) < 2 or new == txt:
                continue
            pert_texts.append(new)
            used.append(tok)
        if not pert_texts:
            continue
        Zp = eng._text_features(pert_texts)
        Ip = np.repeat(I[si:si + 1], len(pert_texts), axis=0)
        mtp = np.ones(len(pert_texts), np.float32)
        mip = np.ones(len(pert_texts), np.float32)
        pp = predict_gated(eng, Zp, Ip, mtp, mip)
        cls = int(base_idx[si])
        imp = [(tok, float(base_p[si, cls] - pp[j, cls])) for j, tok in enumerate(used)]
        imp.sort(key=lambda x: x[1], reverse=True)
        samples.append({
            "si": int(si), "text": txt, "tokens": used,
            "importance": imp, "base_p": float(base_p[si, cls]), "cls": cls,
        })
    print(f"  文本删除：{len(samples)} 条样本进入实验")

    rows = []
    rng = np.random.default_rng(SEED + 7)
    for k in [1, 2, 3, 5]:
        top_texts, top_idx = [], []
        rand_sets = []  # (draw, text, si)
        for s in samples:
            toks = [t for t, _ in s["importance"]][:k]
            if len(toks) < k:
                continue
            new_top = _remove_tokens(s["text"], toks)
            if len(new_top.strip()) < 2:
                continue
            top_texts.append(new_top)
            top_idx.append(s["si"])
            pool = s["tokens"]
            for draw in range(3):
                pick = rng.choice(pool, size=min(k, len(pool)), replace=False).tolist()
                new_rand = _remove_tokens(s["text"], pick)
                if len(new_rand.strip()) < 2:
                    continue
                rand_sets.append((draw, new_rand, s["si"]))
        if not top_texts:
            continue
        Ztop = eng._text_features(top_texts)
        Itop = np.stack([I[si] for si in top_idx]).astype(np.float32)
        ptop = predict_gated(eng, Ztop, Itop,
                             np.ones(len(top_texts), np.float32),
                             np.ones(len(top_texts), np.float32))
        ytop = y[np.array(top_idx)]
        top_m = metric(ytop, ptop)
        top_drop = float(np.mean([
            base_p[si, int(base_p[si].argmax())] - ptop[j, int(base_p[si].argmax())]
            for j, si in enumerate(top_idx)
        ]))
        rand_metrics = []
        for draw in range(3):
            idxs = [j for j, (d, _, _) in enumerate(rand_sets) if d == draw]
            if not idxs:
                continue
            texts = [rand_sets[j][1] for j in idxs]
            sis = [rand_sets[j][2] for j in idxs]
            Zp = eng._text_features(texts)
            Ip = np.stack([I[si] for si in sis]).astype(np.float32)
            pp = predict_gated(eng, Zp, Ip, np.ones(len(texts), np.float32),
                               np.ones(len(texts), np.float32))
            m = metric(y[np.array(sis)], pp)
            drop = float(np.mean([
                base_p[si, int(base_p[si].argmax())] - pp[j, int(base_p[si].argmax())]
                for j, si in enumerate(sis)
            ]))
            rand_metrics.append((m, drop))
        rows.append({
            "modality": "text", "method": "top", "k": k, "n_samples": len(top_idx),
            "accuracy": top_m["accuracy"], "macro_f1": top_m["macro_f1"],
            "mean_prob_drop": round(top_drop, 6), "std_prob_drop": 0.0,
        })
        if rand_metrics:
            rows.append({
                "modality": "text", "method": "random", "k": k,
                "n_samples": int(np.mean([m["n"] for m, _ in rand_metrics])),
                "accuracy": round(float(np.mean([m["accuracy"] for m, _ in rand_metrics])), 6),
                "macro_f1": round(float(np.mean([m["macro_f1"] for m, _ in rand_metrics])), 6),
                "mean_prob_drop": round(float(np.mean([d for _, d in rand_metrics])), 6),
                "std_prob_drop": round(float(np.std([d for _, d in rand_metrics])), 6),
            })
        print(f"  text k={k}: top drop={rows[-2]['mean_prob_drop']:.4f} / "
              f"random drop={rows[-1]['mean_prob_drop']:.4f}")
    return {
        "n_candidates": len(samples),
        "selection": "从完整模态预测正确的测试样本中按类别抽样，用于删除实验。",
        "rows": rows,
        "examples": [{
            "text": s["text"][:80],
            "top_tokens": [{"token": t, "prob_drop": round(d, 6)} for t, d in s["importance"][:3]],
        } for s in samples[:5]],
    }


def _region_features(eng, im):
    w, h = im.size
    cw, ch = max(1, w // 3), max(1, h // 3)
    feats = []
    for rr in range(3):
        for cc in range(3):
            im2 = im.copy()
            draw = ImageDraw.Draw(im2)
            x0, y0 = cc * cw, rr * ch
            x1 = w if cc == 2 else (cc + 1) * cw
            y1 = h if rr == 2 else (rr + 1) * ch
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(128, 128, 128))
            rvec = eng.encoder.encode_pil(im2).astype(np.float32)
            rvec = rvec / max(float(np.linalg.norm(rvec)), 1e-6)
            st = image_stats(im2)
            svec = eng.stats_scaler.transform(
                np.array([[st[c] for c in eng.stat_cols]], dtype=np.float32))[0] * 0.5
            feats.append(np.concatenate([rvec, svec]).astype(np.float32))
    return np.stack(feats)


def _gray_regions(im, regions):
    w, h = im.size
    cw, ch = max(1, w // 3), max(1, h // 3)
    im2 = im.copy()
    draw = ImageDraw.Draw(im2)
    for rr, cc in regions:
        x0, y0 = cc * cw, rr * ch
        x1 = w if cc == 2 else (cc + 1) * cw
        y1 = h if rr == 2 else (rr + 1) * ch
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(128, 128, 128))
    return im2


def _encode_one(eng, im):
    rvec = eng.encoder.encode_pil(im).astype(np.float32)
    rvec = rvec / max(float(np.linalg.norm(rvec)), 1e-6)
    st = image_stats(im)
    svec = eng.stats_scaler.transform(
        np.array([[st[c] for c in eng.stat_cols]], dtype=np.float32))[0] * 0.5
    return np.concatenate([rvec, svec]).astype(np.float32)


def image_deletion_experiment(eng, df, test_df, Z, I, y, n_total: int):
    base_p = predict_gated(eng, Z, I, np.zeros(len(y), np.float32),
                           np.ones(len(y), np.float32))
    pred = CLASSES[base_p.argmax(1)]
    chosen = []
    rng = np.random.default_rng(SEED + 3)
    per_class = max(1, n_total // 3)
    for lab in CLASSES:
        cand = np.where((y == lab) & (pred == lab))[0]
        if len(cand):
            chosen.extend(rng.choice(cand, size=min(per_class, len(cand)),
                                     replace=False).tolist())
    chosen = sorted(set(chosen))
    samples = []
    for si in chosen:
        row = test_df.iloc[si]
        im = load_image(os.path.join(ROOT, "data", "raw", "images", str(row["image_file"])))
        if im is None:
            continue
        feats = _region_features(eng, im)
        zt = np.repeat(Z[si:si + 1], 9, axis=0)
        pp = predict_gated(eng, zt, feats, np.zeros(9, np.float32), np.ones(9, np.float32))
        cls = int(base_p[si].argmax())
        imp = [(rr, cc, float(base_p[si, cls] - pp[rr * 3 + cc, cls]))
               for rr in range(3) for cc in range(3)]
        imp.sort(key=lambda x: x[2], reverse=True)
        samples.append({"si": int(si), "im": im, "imp": imp, "cls": cls})
    print(f"  图像删除：{len(samples)} 条样本进入实验")

    rows = []
    rng = np.random.default_rng(SEED + 11)
    for k in [1, 2, 3]:
        top_feats, top_idx = [], []
        rand_sets = []
        for s in samples:
            top_regions = [(rr, cc) for rr, cc, _ in s["imp"][:k]]
            top_feats.append(_encode_one(eng, _gray_regions(s["im"], top_regions)))
            top_idx.append(s["si"])
            for draw in range(2):
                pick = rng.choice(9, size=k, replace=False).tolist()
                regions = [(j // 3, j % 3) for j in pick]
                rand_sets.append((draw, _encode_one(eng, _gray_regions(s["im"], regions)), s["si"]))
        zt = np.stack([Z[si] for si in top_idx]).astype(np.float32)
        ptop = predict_gated(eng, zt, np.stack(top_feats),
                             np.zeros(len(top_idx), np.float32),
                             np.ones(len(top_idx), np.float32))
        top_m = metric(y[np.array(top_idx)], ptop)
        top_drop = float(np.mean([
            base_p[si, int(base_p[si].argmax())] - ptop[j, int(base_p[si].argmax())]
            for j, si in enumerate(top_idx)
        ]))
        rand_metrics = []
        for draw in range(2):
            items = [(feat, si) for d, feat, si in rand_sets if d == draw]
            if not items:
                continue
            feats = np.stack([x[0] for x in items])
            sis = [x[1] for x in items]
            zt2 = np.stack([Z[si] for si in sis]).astype(np.float32)
            pp = predict_gated(eng, zt2, feats, np.zeros(len(sis), np.float32),
                               np.ones(len(sis), np.float32))
            m = metric(y[np.array(sis)], pp)
            drop = float(np.mean([
                base_p[si, int(base_p[si].argmax())] - pp[j, int(base_p[si].argmax())]
                for j, si in enumerate(sis)
            ]))
            rand_metrics.append((m, drop))
        rows.append({
            "modality": "image", "method": "top", "k": k, "n_samples": len(top_idx),
            "accuracy": top_m["accuracy"], "macro_f1": top_m["macro_f1"],
            "mean_prob_drop": round(top_drop, 6), "std_prob_drop": 0.0,
        })
        if rand_metrics:
            rows.append({
                "modality": "image", "method": "random", "k": k,
                "n_samples": int(np.mean([m["n"] for m, _ in rand_metrics])),
                "accuracy": round(float(np.mean([m["accuracy"] for m, _ in rand_metrics])), 6),
                "macro_f1": round(float(np.mean([m["macro_f1"] for m, _ in rand_metrics])), 6),
                "mean_prob_drop": round(float(np.mean([d for _, d in rand_metrics])), 6),
                "std_prob_drop": round(float(np.std([d for _, d in rand_metrics])), 6),
            })
        print(f"  image k={k}: top drop={rows[-2]['mean_prob_drop']:.4f} / "
              f"random drop={rows[-1]['mean_prob_drop']:.4f}")
    return {
        "n_candidates": len(samples),
        "selection": "从仅图像场景预测正确的测试样本中按类别抽样；遮挡区域用灰色填充。",
        "rows": rows,
    }


def plot_figures(missing, deletion):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.28, "grid.linestyle": ":",
                         "figure.dpi": 150, "savefig.dpi": 150})

    colors = {"text": "#1f77b4", "image": "#ff7f0e", "both": "#d9534f"}
    names = {"text": "仅文本缺失", "image": "仅图像缺失", "both": "双模态缺失"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for model_name, ls in [("gated", "-"), ("early_mlp", "--")]:
        for modality in ["text", "image", "both"]:
            rows = [r for r in missing["rows"]
                    if r["model"] == model_name and r["modality"] == modality]
            xs = [r["ratio"] * 100 for r in rows]
            ys = [r["macro_f1_mean"] for r in rows]
            ax.plot(xs, ys, ls, marker="o", ms=4, color=colors[modality],
                    label=f"{'门控' if model_name == 'gated' else '早期 MLP'}·{names[modality]}")
    ax.set_xlabel("缺失比例（%）")
    ax.set_ylabel("Macro-F1")
    ax.set_title("E2 缺失比例—性能曲线（测试集 1,800 条，3 个随机种子均值）")
    ax.set_xticks([10, 30, 50, 70])
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    path = os.path.join(DOCS, "figures", "fig11_missing_ratio.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, modality, title in [(axes[0], "text", "文本 token 删除"),
                                (axes[1], "image", "图像 3×3 区域删除")]:
        rows = [r for r in deletion[modality]["rows"]]
        for method, color, label in [("top", "#d9534f", "按重要性删除"),
                                     ("random", "#1f77b4", "随机删除")]:
            rs = [r for r in rows if r["method"] == method]
            if not rs:
                continue
            ax.plot([r["k"] for r in rs], [r["mean_prob_drop"] for r in rs],
                    marker="o", color=color, label=label)
        ax.set_xlabel("删除数量 k")
        ax.set_ylabel("真实类概率平均下降")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("E4 删除实验：按重要性删除 vs 随机删除", fontsize=11)
    fig.tight_layout()
    path = os.path.join(DOCS, "figures", "fig12_deletion.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="缺失比例/位置与删除实验")
    ap.add_argument("--text-n", type=int, default=120)
    ap.add_argument("--image-n", type=int, default=18)
    ap.add_argument("--skip-image-deletion", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 66)
    print("12 缺失比例/位置曲线 + 解释删除实验")
    print("=" * 66)
    eng, df, test_df, Z, I, y, early, image_all, resnet, stats_scaled = load_context()
    print(f"  测试集 {len(y):,} 条；文本 {Z.shape[1]} 维；图像 {I.shape[1]} 维")

    print("\n[1] E2 缺失比例曲线")
    missing = missing_ratio_experiment(eng, early, Z, I, y)
    print("\n[2] E2 缺失位置代理实验")
    position = missing_position_experiment(eng, early, Z, I, y)
    print("\n[3] E4 文本删除实验")
    text_del = text_deletion_experiment(eng, df, test_df, Z, I, y, args.text_n)
    if args.skip_image_deletion:
        image_del = {"n_candidates": 0, "rows": [], "selection": "跳过"}
    else:
        print("\n[4] E4 图像区域删除实验")
        image_del = image_deletion_experiment(eng, df, test_df, Z, I, y, args.image_n)
    deletion = {"text": text_del, "image": image_del}

    os.makedirs(TABLES, exist_ok=True)
    pd.DataFrame(missing["rows"]).to_csv(
        os.path.join(TABLES, "缺失比例曲线.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(position["rows"]).to_csv(
        os.path.join(TABLES, "缺失位置曲线.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(deletion["text"]["rows"] + deletion["image"]["rows"]).to_csv(
        os.path.join(TABLES, "删除实验.csv"), index=False, encoding="utf-8-sig")

    result = {
        "generated_at": now_str(),
        "test_n": int(len(y)),
        "missing_ratio": missing,
        "missing_position": position,
        "deletion": deletion,
        "notes": [
            "缺失比例是样本级整模态缺失，不是视频帧级缺失。",
            "缺失位置是特征维连续置零的代理实验，不是真实时间轴位置。",
            "删除实验是敏感性分析，不是因果证明。",
            "所有实验只使用固定 test split，不参与训练或调参。",
        ],
    }
    write_json(os.path.join(OUT, "missing_experiments.json"), result)

    met = read_json(os.path.join(OUT, "models_metrics.json"))
    met["missing_ratio"] = missing
    met["missing_position"] = position
    met["deletion"] = deletion
    met["extra_experiments_generated_at"] = now_str()
    write_json(os.path.join(OUT, "models_metrics.json"), met)
    exp = read_json(os.path.join(OUT, "explainability.json"))
    exp["missing_ratio"] = missing
    exp["deletion_text"] = text_del
    exp["deletion_image"] = image_del
    exp["extra_experiments_generated_at"] = now_str()
    write_json(os.path.join(OUT, "explainability.json"), exp)

    if not args.skip_figures:
        print("\n[5] 生成 E2/E4 插图")
        plot_figures(missing, deletion)

    print(f"\n[完成] 总耗时 {time.time() - t0:.1f}s")
    print(f"  事实表：output/missing_experiments.json")
    print(f"  底表：output/tables/缺失比例曲线.csv / 缺失位置曲线.csv / 删除实验.csv")


if __name__ == "__main__":
    main()
