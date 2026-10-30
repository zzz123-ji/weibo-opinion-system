# -*- coding: utf-8 -*-
"""04 多模态情感建模：门控融合 + 模态缺失鲁棒性 + 可解释性。

模型
----
- 文本塔：字符级 TF-IDF（2~4 gram）→ TruncatedSVD 256 维
- 图像塔：ResNet18 冻结特征（L2 归一化）+ 7 维图像质量特征
- 融合：门控网络（按样本学习文本/图像权重）+ 模态 dropout
- 对照：text-only、image-only、early fusion（LR / MLP）、late fusion、等权融合

评估
----
- 完整模态、仅文本、仅图像、双缺失
- 真实无图外部测试集（3,000 条）
- bootstrap 95% 置信区间、混淆矩阵、分类报告
- 门控权重、文本 token 遮挡、图像 3×3 区域遮挡

输出
----
output/models_metrics.json
output/explainability.json
output/models/*.joblib / *.pt
output/tables/模型对比.csv
output/tables/缺失场景鲁棒性.csv
output/tables/外部测试.csv
output/tables/混淆矩阵.csv
output/tables/门控权重.csv
output/tables/文本遮挡解释.csv
output/tables/图像遮挡解释.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import ImageDraw
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd  # noqa: F401  (确保依赖可用)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import (LABEL_NAMES, LABEL_ORDER, MODELS, OUT, PROCESSED, ROOT,
                        TABLES, now_str, write_json)
from lib_image import ImageEncoder, image_stats, load_image
from lib_text import tokenize_keywords

warnings.filterwarnings("ignore")
SEED = 42
CLASSES = np.array([-1, 0, 1])
STAT_COLS = ["brightness", "saturation", "contrast", "colorfulness",
             "edge_density", "entropy", "warm_ratio"]
N_BOOT = 1000


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) - 1)))


def py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return py(v.tolist())
    if isinstance(v, dict):
        return {str(k): py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [py(x) for x in v]
    return v


def metrics_bundle(y_true, y_pred, y_prob=None) -> dict:
    rep = classification_report(y_true, y_pred, labels=list(CLASSES),
                                output_dict=True, zero_division=0)
    per_class = {}
    for lab in CLASSES:
        r = rep.get(str(lab), {})
        per_class[str(lab)] = {
            "label_name": LABEL_NAMES[int(lab)],
            "precision": round(float(r.get("precision", 0.0)), 6),
            "recall": round(float(r.get("recall", 0.0)), 6),
            "f1": round(float(r.get("f1-score", 0.0)), 6),
            "support": int(r.get("support", 0)),
        }
    out = {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 6),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(CLASSES)).tolist(),
    }
    if y_prob is not None:
        try:
            out["log_loss"] = round(float(log_loss(y_true, y_prob, labels=list(CLASSES))), 6)
        except Exception:
            out["log_loss"] = None
    return out


def bootstrap_macro_f1(y_true, y_pred, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    vals = np.asarray(vals)
    return {
        "mean": round(float(vals.mean()), 6),
        "low": round(float(np.percentile(vals, 2.5)), 6),
        "high": round(float(np.percentile(vals, 97.5)), 6),
        "n_boot": int(n_boot),
    }


def transform_text(texts, vectorizer, svd, scaler) -> np.ndarray:
    x = vectorizer.transform(pd.Series(texts).fillna("").astype(str).tolist())
    z = svd.transform(x)
    return scaler.transform(z).astype(np.float32)


# ---------------------------------------------------------------------------
# 神经网络
# ---------------------------------------------------------------------------
class GatedFusionNet(nn.Module):
    def __init__(self, d_text=256, d_image=519, hidden=192, dropout=0.25):
        super().__init__()
        self.text_tower = nn.Sequential(
            nn.Linear(d_text, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, hidden), nn.ReLU(),
        )
        self.text_skip = nn.Linear(d_text, hidden)
        self.image_tower = nn.Sequential(
            nn.Linear(d_image, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, hidden), nn.ReLU(),
        )
        self.image_skip = nn.Linear(d_image, hidden)
        self.missing_text = nn.Parameter(torch.zeros(hidden))
        self.missing_image = nn.Parameter(torch.zeros(hidden))
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2 + 2, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 96), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(96, 3),
        )
        # 单模态 logit 残差：给每条模态保留直接判别通路，避免联合训练时被另一模态淹没
        self.text_logit = nn.Linear(d_text, 3)
        self.image_logit = nn.Linear(d_image, 3)

    def forward(self, zt, zi, mt, mi, force_equal=False):
        ht_raw = self.text_tower(zt) + self.text_skip(zt)
        hi_raw = self.image_tower(zi) + self.image_skip(zi)
        mt = mt.view(-1, 1)
        mi = mi.view(-1, 1)
        ht = mt * ht_raw + (1.0 - mt) * self.missing_text
        hi = mi * hi_raw + (1.0 - mi) * self.missing_image
        avail = torch.cat([mt, mi], dim=1)
        g_logits = self.gate(torch.cat([ht, hi, mt, mi], dim=1))
        g_logits = g_logits.masked_fill(avail < 0.5, -1e9)
        g = torch.softmax(g_logits, dim=1)
        both_missing = (mt + mi) < 0.5
        g = torch.where(both_missing, torch.full_like(g, 0.5), g)
        if force_equal:
            g = torch.where(avail > 0.5, torch.ones_like(g), torch.zeros_like(g))
            g = g / g.sum(dim=1, keepdim=True).clamp_min(1e-9)
        fused = g[:, 0:1] * ht + g[:, 1:2] * hi
        aux_t = self.text_logit(zt)
        aux_i = self.image_logit(zi)
        logits = self.classifier(fused) + mt * aux_t + mi * aux_i
        return logits, g, aux_t, aux_i


class EarlyFusionMLP(nn.Module):
    """无门控对照：直接拼接两模态 + 缺失指示，零填充缺失模态。"""

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


def _batch_masks(n, drop_text=0.0, drop_image=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    mt = (rng.random(n) >= drop_text).astype(np.float32)
    mi = (rng.random(n) >= drop_image).astype(np.float32)
    return torch.from_numpy(mt), torch.from_numpy(mi)


def train_torch_model(kind: str, Ztr, Itr, ytr, Zva, Iva, yva,
                      epochs: int = 35, batch_size: int = 128,
                      patience: int = 8) -> tuple[nn.Module, dict]:
    set_seed(SEED)
    if kind == "gated":
        model = GatedFusionNet()
    else:
        model = EarlyFusionMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    rng = np.random.default_rng(SEED)
    n = len(ytr)
    yt = torch.from_numpy((ytr + 1).astype(np.int64))
    Ztr_t = torch.from_numpy(Ztr)
    Itr_t = torch.from_numpy(Itr)
    Zva_t = torch.from_numpy(Zva)
    Iva_t = torch.from_numpy(Iva)
    best_f1, best_state, bad = -1.0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            zt = Ztr_t[idx]
            zi = Itr_t[idx]
            yy = yt[idx]
            mt, mi = _batch_masks(len(idx), 0.15, 0.15, rng)
            out = model(zt, zi, mt, mi)
            if kind == "gated":
                logits, g, aux_t, aux_i = out
                loss = loss_fn(logits, yy)
                mt_flat = mt.view(-1) > 0.5
                mi_flat = mi.view(-1) > 0.5
                if mt_flat.any():
                    loss = loss + 0.3 * loss_fn(aux_t[mt_flat], yy[mt_flat])
                if mi_flat.any():
                    loss = loss + 0.3 * loss_fn(aux_i[mi_flat], yy[mi_flat])
            else:
                logits, _ = out
                loss = loss_fn(logits, yy)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.item()) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            out = model(Zva_t, Iva_t, torch.ones(len(yva)), torch.ones(len(yva)))
            logits = out[0]
            pred = logits.argmax(dim=1).numpy()
        val_f1 = f1_score(yva + 1, pred, average="macro")
        history.append({"epoch": epoch, "train_loss": round(total_loss / n, 6),
                        "val_macro_f1": round(float(val_f1), 6)})
        if val_f1 > best_f1 + 1e-5:
            best_f1 = float(val_f1)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_macro_f1": round(best_f1, 6), "history": history,
                   "epochs_ran": len(history)}


def predict_torch(model, Z, I, mt, mi, batch_size=512, force_equal=False):
    model.eval()
    probs, gates = [], []
    with torch.no_grad():
        for start in range(0, len(Z), batch_size):
            zt = torch.from_numpy(Z[start:start + batch_size])
            zi = torch.from_numpy(I[start:start + batch_size])
            m1 = torch.from_numpy(mt[start:start + batch_size].astype(np.float32))
            m2 = torch.from_numpy(mi[start:start + batch_size].astype(np.float32))
            logits, g = model(zt, zi, m1, m2, force_equal=force_equal)[:2]
            probs.append(torch.softmax(logits, dim=1).numpy())
            gates.append(g.numpy())
    return np.concatenate(probs), np.concatenate(gates)


def scenario_masks(name: str, n: int):
    if name == "complete":
        return np.ones(n, np.float32), np.ones(n, np.float32)
    if name == "text-only":
        return np.ones(n, np.float32), np.zeros(n, np.float32)
    if name == "image-only":
        return np.zeros(n, np.float32), np.ones(n, np.float32)
    if name == "both-missing":
        return np.zeros(n, np.float32), np.zeros(n, np.float32)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--skip-occlusion", action="store_true")
    ap.add_argument("--occlusion-text-n", type=int, default=120)
    ap.add_argument("--occlusion-image-n", type=int, default=45)
    args = ap.parse_args()

    t0 = time.time()
    set_seed(SEED)
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)
    print("=" * 66)
    print("04 多模态情感建模：门控融合 / 缺失鲁棒 / 可解释")
    print("=" * 66)

    # ---------- 数据 ----------
    df = pd.read_csv(os.path.join(PROCESSED, "posts_clean.csv"), dtype=str)
    df["label"] = df["label"].astype(int)
    df["text_clean"] = df["text_clean"].fillna("")
    df["image_row"] = df["image_row"].astype(int)
    resnet = np.load(os.path.join(PROCESSED, "image_features.npy")).astype(np.float32)
    # L2 归一化，避免 ResNet 特征幅值影响融合
    resnet = resnet / np.clip(np.linalg.norm(resnet, axis=1, keepdims=True), 1e-6, None)

    stats_df = pd.read_csv(os.path.join(PROCESSED, "image_stats.csv")).set_index("post_hash")
    stats_aligned = stats_df.loc[df["post_hash"], STAT_COLS].to_numpy(np.float32)
    stats_scaler = StandardScaler().fit(stats_aligned[(df["split"] == "train").to_numpy()])
    stats_scaled = (stats_scaler.transform(stats_aligned) * 0.5).astype(np.float32)
    image_all = np.concatenate([resnet, stats_scaled], axis=1).astype(np.float32)

    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    idx_tr = np.where(train_mask)[0]
    idx_va = np.where(val_mask)[0]
    idx_te = np.where(test_mask)[0]
    ytr = df.loc[train_mask, "label"].to_numpy()
    yva = df.loc[val_mask, "label"].to_numpy()
    yte = df.loc[test_mask, "label"].to_numpy()
    print(f"  训练 {len(idx_tr):,} / 验证 {len(idx_va):,} / 测试 {len(idx_te):,}；"
          f"图像特征 {image_all.shape[1]} 维（ResNet 512 + 统计 7）")

    # ---------- 文本特征 ----------
    print("\n[1] 文本 TF-IDF + SVD")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                 min_df=3, max_features=80000,
                                 sublinear_tf=True, dtype=np.float32)
    Xtr = vectorizer.fit_transform(df.loc[train_mask, "text_clean"].tolist())
    svd = TruncatedSVD(n_components=256, random_state=SEED, n_iter=6)
    Ztr_raw = svd.fit_transform(Xtr)
    text_scaler = StandardScaler().fit(Ztr_raw)
    Ztr = text_scaler.transform(Ztr_raw).astype(np.float32)
    Zva = transform_text(df.loc[val_mask, "text_clean"], vectorizer, svd, text_scaler)
    Zte = transform_text(df.loc[test_mask, "text_clean"], vectorizer, svd, text_scaler)
    print(f"  TF-IDF {Xtr.shape[1]:,} 维 → SVD 256 维；"
          f"解释方差 {svd.explained_variance_ratio_.sum():.3f}")
    del Xtr, Ztr_raw

    Itr, Iva, Ite = image_all[idx_tr], image_all[idx_va], image_all[idx_te]

    # ---------- 线性对照 ----------
    print("\n[2] 线性对照模型")
    text_lr = LogisticRegression(max_iter=3000, C=2.0, random_state=SEED)
    text_lr.fit(Ztr, ytr)
    image_lr = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED)
    image_lr.fit(Itr, ytr)
    early_lr = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED)
    early_lr.fit(np.concatenate([Ztr, Itr], axis=1), ytr)
    print("  text / image / early LR 训练完成")

    # late fusion 权重在验证集上选
    pt_va = text_lr.predict_proba(Zva)
    pi_va = image_lr.predict_proba(Iva)
    best_alpha, best_alpha_f1 = 0.5, -1.0
    for a in np.linspace(0, 1, 21):
        p = a * pt_va + (1 - a) * pi_va
        f = f1_score(yva, CLASSES[p.argmax(1)], average="macro")
        if f > best_alpha_f1:
            best_alpha, best_alpha_f1 = float(a), float(f)
    print(f"  late fusion alpha={best_alpha:.2f}（验证集 macro-F1 {best_alpha_f1:.4f}）")

    # ---------- 神经网络 ----------
    print("\n[3] 门控融合网络（模态 dropout 15%）")
    gated, gated_hist = train_torch_model("gated", Ztr, Itr, ytr, Zva, Iva, yva,
                                           epochs=args.epochs)
    print(f"  最优验证 macro-F1 {gated_hist['best_val_macro_f1']:.4f}，"
          f"训练 {gated_hist['epochs_ran']} 轮")

    print("\n[4] 早期融合 MLP 对照（无门控）")
    early_mlp, early_hist = train_torch_model("early", Ztr, Itr, ytr, Zva, Iva, yva,
                                              epochs=args.epochs)
    print(f"  最优验证 macro-F1 {early_hist['best_val_macro_f1']:.4f}，"
          f"训练 {early_hist['epochs_ran']} 轮")

    # ---------- 测试集评估 ----------
    print("\n[5] 测试集：完整模态 + 缺失场景")
    n_te = len(yte)
    scenario_probs = {}
    scenario_gates = {}

    # 线性模型
    pt_te = text_lr.predict_proba(Zte)
    pi_te = image_lr.predict_proba(Ite)
    scenario_probs[("text_lr", "complete")] = pt_te
    scenario_probs[("image_lr", "complete")] = pi_te
    scenario_probs[("early_lr", "complete")] = early_lr.predict_proba(
        np.concatenate([Zte, Ite], axis=1))
    scenario_probs[("late_lr", "complete")] = best_alpha * pt_te + (1 - best_alpha) * pi_te
    prior = np.full((n_te, 3), 1 / 3, dtype=np.float32)

    for sc in ["text-only", "image-only", "both-missing"]:
        mt, mi = scenario_masks(sc, n_te)
        zt = Zte * mt[:, None]
        zi = Ite * mi[:, None]
        scenario_probs[("early_lr", sc)] = early_lr.predict_proba(
            np.concatenate([zt, zi], axis=1))
        if sc == "text-only":
            scenario_probs[("late_lr", sc)] = pt_te
        elif sc == "image-only":
            scenario_probs[("late_lr", sc)] = pi_te
        else:
            scenario_probs[("late_lr", sc)] = prior.copy()

    # 神经网络
    for model_name, model in [("gated", gated), ("early_mlp", early_mlp)]:
        for sc in ["complete", "text-only", "image-only", "both-missing"]:
            mt, mi = scenario_masks(sc, n_te)
            p, g = predict_torch(model, Zte, Ite, mt, mi)
            scenario_probs[(model_name, sc)] = p
            scenario_gates[(model_name, sc)] = g
    # 等权融合消融（复用门控模型的两个塔）
    p_eq, g_eq = predict_torch(gated, Zte, Ite, np.ones(n_te), np.ones(n_te),
                               force_equal=True)
    scenario_probs[("gated_equal", "complete")] = p_eq
    scenario_gates[("gated_equal", "complete")] = g_eq

    model_labels = {
        "text_lr": "文本单模态（TF-IDF+SVD+LR）",
        "image_lr": "图像单模态（ResNet+统计+LR）",
        "early_lr": "早期融合（特征拼接+LR）",
        "late_lr": "晚期融合（概率加权）",
        "gated": "门控融合（本文）",
        "gated_equal": "门控融合-等权消融",
        "early_mlp": "早期融合 MLP（无门控对照）",
    }

    complete_metrics = {}
    for name in ["text_lr", "image_lr", "early_lr", "late_lr", "gated", "gated_equal", "early_mlp"]:
        p = scenario_probs[(name, "complete")]
        pred = CLASSES[p.argmax(1)]
        complete_metrics[name] = metrics_bundle(yte, pred, p)
        complete_metrics[name]["bootstrap_macro_f1"] = bootstrap_macro_f1(yte, pred)
        print(f"  {model_labels[name]:<32} acc={complete_metrics[name]['accuracy']:.4f} "
              f"macro-F1={complete_metrics[name]['macro_f1']:.4f} "
              f"[{complete_metrics[name]['bootstrap_macro_f1']['low']:.4f},"
              f"{complete_metrics[name]['bootstrap_macro_f1']['high']:.4f}]")

    robustness = []
    for name in ["early_lr", "late_lr", "gated", "early_mlp"]:
        for sc in ["text-only", "image-only", "both-missing"]:
            p = scenario_probs[(name, sc)]
            pred = CLASSES[p.argmax(1)]
            m = metrics_bundle(yte, pred, p)
            robustness.append({"model": name, "model_label": model_labels[name],
                               "scenario": sc, **{k: m[k] for k in
                                                  ["n", "accuracy", "macro_f1", "weighted_f1"]}})
            print(f"  {model_labels[name]:<32} {sc:<13} acc={m['accuracy']:.4f} "
                  f"macro-F1={m['macro_f1']:.4f}")

    # ---------- 外部无图测试 ----------
    print("\n[6] 外部无图测试集（3,000 条）")
    ext = pd.read_csv(os.path.join(PROCESSED, "external_test_clean.csv"), dtype=str)
    ext["label"] = ext["label"].astype(int)
    yext = ext["label"].to_numpy()
    Zext = transform_text(ext["text_clean"].fillna(""), vectorizer, svd, text_scaler)
    Iext_zero = np.zeros((len(ext), image_all.shape[1]), dtype=np.float32)
    mt1 = np.ones(len(ext), np.float32)
    mi0 = np.zeros(len(ext), np.float32)

    ext_probs = {
        "text_lr": text_lr.predict_proba(Zext),
        "early_lr": early_lr.predict_proba(np.concatenate([Zext, Iext_zero], axis=1)),
        "late_lr": text_lr.predict_proba(Zext),
        "gated": predict_torch(gated, Zext, Iext_zero, mt1, mi0)[0],
        "early_mlp": predict_torch(early_mlp, Zext, Iext_zero, mt1, mi0)[0],
    }
    external_metrics = {}
    for name, p in ext_probs.items():
        pred = CLASSES[p.argmax(1)]
        external_metrics[name] = metrics_bundle(yext, pred, p)
        external_metrics[name]["bootstrap_macro_f1"] = bootstrap_macro_f1(yext, pred)
        print(f"  {model_labels[name]:<32} acc={external_metrics[name]['accuracy']:.4f} "
              f"macro-F1={external_metrics[name]['macro_f1']:.4f}")

    # ---------- 门控权重 ----------
    print("\n[7] 门控权重")
    gate_test = scenario_gates[("gated", "complete")]
    pred_te = CLASSES[scenario_probs[("gated", "complete")].argmax(1)]
    gate_rows = [{
        "group": "all", "label": "全部", "n": int(n_te),
        "text_gate": round(float(gate_test[:, 0].mean()), 6),
        "image_gate": round(float(gate_test[:, 1].mean()), 6),
    }]
    for lab in CLASSES:
        m = yte == lab
        gate_rows.append({
            "group": "true_label", "label": LABEL_NAMES[int(lab)], "n": int(m.sum()),
            "text_gate": round(float(gate_test[m, 0].mean()), 6),
            "image_gate": round(float(gate_test[m, 1].mean()), 6),
        })
    for lab in CLASSES:
        m = pred_te == lab
        gate_rows.append({
            "group": "pred_label", "label": LABEL_NAMES[int(lab)], "n": int(m.sum()),
            "text_gate": round(float(gate_test[m, 0].mean()), 6),
            "image_gate": round(float(gate_test[m, 1].mean()), 6),
        })
    pd.DataFrame(gate_rows).to_csv(os.path.join(TABLES, "门控权重.csv"),
                                   index=False, encoding="utf-8-sig")
    print("  文本门控均值 %.3f，图像门控均值 %.3f" % (
        gate_test[:, 0].mean(), gate_test[:, 1].mean()))

    # ---------- 文本遮挡 ----------
    text_occ = {"n_samples": 0, "top_tokens": [], "examples": []}
    if not args.skip_occlusion:
        print("\n[8] 文本 token 遮挡解释")
        rng = np.random.default_rng(SEED)
        # 每类取预测正确的样本，保证覆盖三类
        chosen = []
        per_class = max(1, args.occlusion_text_n // 3)
        for lab in CLASSES:
            cand = np.where((yte == lab) & (pred_te == lab))[0]
            if len(cand) == 0:
                continue
            take = min(per_class, len(cand))
            chosen.extend(rng.choice(cand, size=take, replace=False).tolist())
        chosen = sorted(set(chosen))
        base_p = scenario_probs[("gated", "complete")][chosen]
        base_pred = base_p.argmax(1)
        texts, pert_texts, meta = [], [], []
        for si in chosen:
            txt = str(df.iloc[idx_te[si]]["text_clean"])
            toks = []
            for w in tokenize_keywords(txt):
                if len(w) >= 2 and w in txt and w not in toks:
                    toks.append(w)
            toks = toks[:8]
            for tok in toks:
                new = txt.replace(tok, "")
                if len(new.strip()) < 4 or new == txt:
                    continue
                pert_texts.append(new)
                meta.append((si, tok))
        if pert_texts:
            Zp = transform_text(pert_texts, vectorizer, svd, text_scaler)
            # 每条扰动复用原样本图像
            Ip = np.stack([Ite[m[0]] for m in meta]).astype(np.float32)
            mtp = np.ones(len(meta), np.float32)
            mip = np.ones(len(meta), np.float32)
            pp, _ = predict_torch(gated, Zp, Ip, mtp, mip)
            drops = []
            for j, (si, tok) in enumerate(meta):
                base_cls = int(base_pred[chosen.index(si)])
                drop = float(base_p[chosen.index(si), base_cls] - pp[j, base_cls])
                drops.append((tok, drop, si, pp[j].argmax()))
            tok_stats = {}
            for tok, drop, si, new_pred in drops:
                tok_stats.setdefault(tok, []).append(drop)
            rows = []
            for tok, vals in tok_stats.items():
                if len(vals) < 3:
                    continue
                rows.append({"token": tok, "n": len(vals),
                             "mean_prob_drop": round(float(np.mean(vals)), 6),
                             "max_prob_drop": round(float(np.max(vals)), 6)})
            rows = sorted(rows, key=lambda x: x["mean_prob_drop"], reverse=True)[:25]
            pd.DataFrame(rows).to_csv(os.path.join(TABLES, "文本遮挡解释.csv"),
                                      index=False, encoding="utf-8-sig")
            # 每条样本影响最大的 token
            by_sample = {}
            for tok, drop, si, new_pred in drops:
                if si not in by_sample or drop > by_sample[si][1]:
                    by_sample[si] = (tok, drop, int(new_pred))
            examples = []
            for si, (tok, drop, new_pred) in list(by_sample.items())[:20]:
                examples.append({
                    "text": str(df.iloc[idx_te[si]]["text_clean"])[:80],
                    "true_label": LABEL_NAMES[int(yte[si])],
                    "top_token": tok,
                    "prob_drop": round(float(drop), 6),
                    "flip": bool(new_pred != int(base_pred[chosen.index(si)])),
                })
            text_occ = {
                "n_samples": int(len(chosen)),
                "n_perturbations": int(len(meta)),
                "top_tokens": rows,
                "examples": examples,
                "method": "删除单个 jieba 关键词后重算 TF-IDF/SVD，观察真实类概率下降",
            }
            print("  遮挡 token top5：" + "、".join(
                f"{r['token']}({r['mean_prob_drop']:+.3f})" for r in rows[:5]))

    # ---------- 图像遮挡 ----------
    image_occ = {"n_samples": 0, "grid": [], "by_class": {}}
    if not args.skip_occlusion:
        print("\n[9] 图像 3×3 区域遮挡解释（仅图像场景）")
        encoder = ImageEncoder("resnet18")
        rng = np.random.default_rng(SEED + 1)
        chosen = []
        per_class = max(1, args.occlusion_image_n // 3)
        pred_img = CLASSES[scenario_probs[("gated", "image-only")].argmax(1)]
        for lab in CLASSES:
            cand = np.where((yte == lab) & (pred_img == lab))[0]
            if len(cand) == 0:
                continue
            take = min(per_class, len(cand))
            chosen.extend(rng.choice(cand, size=take, replace=False).tolist())
        chosen = sorted(set(chosen))
        base_p = scenario_probs[("gated", "image-only")][chosen]
        base_cls = base_p.argmax(1)
        region_drops = {lab: np.zeros((3, 3)) for lab in CLASSES}
        region_counts = {lab: np.zeros((3, 3)) for lab in CLASSES}
        all_drops = np.zeros((3, 3))
        n_done = 0
        for si in chosen:
            row = df.iloc[idx_te[si]]
            img_path = os.path.join(ROOT, "data", "raw", "images", str(row["image_file"]))
            im0 = load_image(img_path)
            if im0 is None:
                continue
            w, h = im0.size
            cw, ch = max(1, w // 3), max(1, h // 3)
            base_resnet = resnet[row["image_row"]]
            base_stat = stats_scaled[row["image_row"]]
            base_feat = np.concatenate([base_resnet, base_stat]).astype(np.float32)
            feats = []
            for rr in range(3):
                for cc in range(3):
                    im = im0.copy()
                    draw = ImageDraw.Draw(im)
                    x0, y0 = cc * cw, rr * ch
                    x1 = w if cc == 2 else (cc + 1) * cw
                    y1 = h if rr == 2 else (rr + 1) * ch
                    draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(128, 128, 128))
                    rvec = encoder.encode_pil(im)
                    rvec = rvec / max(float(np.linalg.norm(rvec)), 1e-6)
                    st = image_stats(im)
                    svec = stats_scaler.transform(
                        np.array([[st[c] for c in STAT_COLS]], dtype=np.float32))[0] * 0.5
                    feats.append(np.concatenate([rvec, svec]).astype(np.float32))
            feats = np.stack(feats)
            zt = np.repeat(Zte[si:si + 1], 9, axis=0)
            mt = np.zeros(9, np.float32)
            mi = np.ones(9, np.float32)
            pp, _ = predict_torch(gated, zt, feats, mt, mi)
            cls = int(base_cls[chosen.index(si)])
            drop = base_p[chosen.index(si), cls] - pp[:, cls]
            all_drops += drop.reshape(3, 3)
            lab = int(yte[si])
            region_drops[lab] += drop.reshape(3, 3)
            region_counts[lab] += 1
            n_done += 1
            if n_done % 15 == 0:
                print(f"    {n_done}/{len(chosen)}")
        if n_done:
            all_drops /= n_done
            grid_rows = []
            for rr in range(3):
                for cc in range(3):
                    grid_rows.append({"row": rr, "col": cc,
                                      "mean_prob_drop": round(float(all_drops[rr, cc]), 6)})
            pd.DataFrame(grid_rows).to_csv(os.path.join(TABLES, "图像遮挡解释.csv"),
                                           index=False, encoding="utf-8-sig")
            by_class = {}
            for lab in CLASSES:
                if float(region_counts[lab].sum()) > 0:
                    m = region_drops[lab] / region_counts[lab]
                    by_class[str(lab)] = [[round(float(x), 6) for x in r] for r in m]
            image_occ = {
                "n_samples": int(n_done),
                "grid": [[round(float(x), 6) for x in r] for r in all_drops],
                "grid_rows": grid_rows,
                "by_class": by_class,
                "method": "在仅图像场景下，将图像划分为 3×3 区域，逐区域填充灰色后重算 ResNet 特征，观察真实类概率下降",
            }
            print("  区域概率下降（3×3）：")
            for r in all_drops:
                print("    " + " ".join(f"{x:+.4f}" for x in r))

    # ---------- 保存模型 ----------
    print("\n[10] 保存模型与结果")
    joblib.dump({
        "vectorizer": vectorizer, "svd": svd, "text_scaler": text_scaler,
        "stats_scaler": stats_scaler, "stat_cols": STAT_COLS,
        "text_lr": text_lr, "image_lr": image_lr, "early_lr": early_lr,
        "late_alpha": best_alpha, "classes": CLASSES.tolist(),
    }, os.path.join(MODELS, "baselines.joblib"), compress=3)
    torch.save({"state_dict": gated.state_dict(),
                "config": {"d_text": 256, "d_image": int(image_all.shape[1]),
                           "hidden": 192, "dropout": 0.25}},
               os.path.join(MODELS, "gated_fusion.pt"))
    torch.save({"state_dict": early_mlp.state_dict(),
                "config": {"d_text": 256, "d_image": int(image_all.shape[1]),
                           "hidden": 256, "dropout": 0.25}},
               os.path.join(MODELS, "early_fusion_mlp.pt"))

    # ---------- 表 ----------
    comp_rows = []
    for name, m in complete_metrics.items():
        comp_rows.append({
            "model": name, "model_label": model_labels[name],
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"], "log_loss": m.get("log_loss"),
            "ci_low": m["bootstrap_macro_f1"]["low"],
            "ci_high": m["bootstrap_macro_f1"]["high"],
        })
    pd.DataFrame(comp_rows).to_csv(os.path.join(TABLES, "模型对比.csv"),
                                   index=False, encoding="utf-8-sig")
    pd.DataFrame(robustness).to_csv(os.path.join(TABLES, "缺失场景鲁棒性.csv"),
                                    index=False, encoding="utf-8-sig")
    ext_rows = []
    for name, m in external_metrics.items():
        ext_rows.append({
            "model": name, "model_label": model_labels[name],
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "ci_low": m["bootstrap_macro_f1"]["low"],
            "ci_high": m["bootstrap_macro_f1"]["high"],
        })
    pd.DataFrame(ext_rows).to_csv(os.path.join(TABLES, "外部测试.csv"),
                                  index=False, encoding="utf-8-sig")

    cm_rows = []
    for name, m in complete_metrics.items():
        cm = m["confusion_matrix"]
        for i, true_lab in enumerate(CLASSES):
            for j, pred_lab in enumerate(CLASSES):
                cm_rows.append({"model": name, "true_label": int(true_lab),
                                "pred_label": int(pred_lab), "count": int(cm[i][j])})
    pd.DataFrame(cm_rows).to_csv(os.path.join(TABLES, "混淆矩阵.csv"),
                                 index=False, encoding="utf-8-sig")

    # ---------- JSON ----------
    result = {
        "generated_at": now_str(),
        "config": {
            "seed": SEED,
            "text_features": "char_wb 2-4 gram TF-IDF (min_df=3, max_features=80000) → SVD 256 → StandardScaler",
            "image_features": "ResNet18 ImageNet 冻结 512 维 L2 归一化 + 7 维图像统计（标准化×0.5）",
            "gated_model": "双塔 MLP + 样本级门控 + 模态 dropout 0.15 + 缺失嵌入",
            "early_mlp": "拼接两模态 + 缺失指示 + 零填充，无门控",
            "training": {"batch_size": 128, "optimizer": "AdamW", "lr": 0.0015,
                         "weight_decay": 0.0001, "label_smoothing": 0.05,
                         "max_epochs": args.epochs, "early_stopping_patience": 8},
            "late_fusion_alpha": best_alpha,
        },
        "data": {
            "train": int(len(idx_tr)), "val": int(len(idx_va)), "test": int(len(idx_te)),
            "external_no_image": int(len(ext)),
            "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
            "image_dim": int(image_all.shape[1]),
            "text_svd_explained_variance": round(float(svd.explained_variance_ratio_.sum()), 6),
        },
        "complete": complete_metrics,
        "robustness": robustness,
        "external_no_image": external_metrics,
        "gates": gate_rows,
        "training_history": {"gated": gated_hist, "early_mlp": early_hist},
        "explainability": {"text_occlusion": text_occ, "image_occlusion": image_occ},
        "notes": [
            "所有指标基于 data/processed/posts_clean.csv 的固定 test split（1,800 条，三类各 600）。",
            "外部无图测试集来自同一公开数据集的文本子集（3,000 条），图像模态真实缺失。",
            "both-missing 场景下模型只能依赖先验/缺失嵌入，指标用于展示下限而非实际可用能力。",
            "图像遮挡在仅图像场景下计算：完整模态下门控主要依赖文本，遮挡图像的概率变化接近 0，属于预期结果。",
        ],
    }
    write_json(os.path.join(OUT, "models_metrics.json"), py(result))
    write_json(os.path.join(OUT, "explainability.json"), py({
        "generated_at": now_str(),
        "text_occlusion": text_occ,
        "image_occlusion": image_occ,
        "gates": gate_rows,
    }))

    # ---------- 模型卡 ----------
    g = complete_metrics["gated"]
    t = complete_metrics["text_lr"]
    e = external_metrics["gated"]
    card = f"""# 模型卡：门控多模态情感分析

## 模型
- 文本塔：字符级 TF-IDF（2~4 gram）→ SVD 256 维 → 标准化
- 图像塔：ResNet18（ImageNet 冻结）512 维 L2 归一化 + 7 维图像统计
- 融合：样本级门控（文本/图像权重）+ 模态 dropout 0.15 + 缺失模态嵌入
- 分类：3 类（负面 / 中性 / 正面）

## 测试集表现（1,800 条，完整模态）
- Accuracy：{g['accuracy']:.4f}
- Macro-F1：{g['macro_f1']:.4f}（95% CI {g['bootstrap_macro_f1']['low']:.4f}~{g['bootstrap_macro_f1']['high']:.4f}）
- 文本单模态 Macro-F1：{t['macro_f1']:.4f}
- 外部无图测试 Macro-F1：{e['macro_f1']:.4f}

## 缺失场景
见 `output/tables/缺失场景鲁棒性.csv`。门控在仅文本/仅图像/双缺失下均有明确输出路径，
双缺失时退化为先验，不作为可用能力宣传。

## 限制
- 标签来自公开数据集的弱标注，存在噪声；
- 图像情感信号弱于文本，门控通常偏向文本；
- 数据为 2020 年疫情早期微博，不能直接外推到其他事件；
- 模型不用于个体用户画像或自动处置，只做聚合舆情辅助。
"""
    with open(os.path.join(MODELS, "MODEL_CARD.md"), "w", encoding="utf-8") as f:
        f.write(card)

    print(f"\n[完成] 建模与解释完成，总耗时 {time.time() - t0:.1f}s")
    print(f"  指标：{os.path.relpath(os.path.join(OUT, 'models_metrics.json'))}")
    print(f"  模型：{os.path.relpath(MODELS)}")


if __name__ == "__main__":
    main()
