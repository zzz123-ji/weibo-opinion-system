# -*- coding: utf-8 -*-
"""06 生成研究报告插图（docs/figures/fig1~fig10）。

所有数值来自 output/*.json、output/tables/*.csv、data/processed/*.csv，
不在绘图脚本里硬编码实验指标。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import (FIGURES, LABEL_NAMES, OUT, PROCESSED, ROOT, TABLES,
                        read_json)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": ":",
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

C_NEG = "#d9534f"
C_NEU = "#8fa3c8"
C_POS = "#2ca87f"
C_MAIN = "#1f77b4"
C_ALT = "#ff7f0e"
C_GATED = "#2ca02c"
C_PURPLE = "#8e6bbf"


def save(fig, name):
    path = os.path.join(FIGURES, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {os.path.relpath(path, ROOT)}")


def load():
    ana = read_json(os.path.join(OUT, "analysis_results.json"))
    met = read_json(os.path.join(OUT, "models_metrics.json"))
    exp = read_json(os.path.join(OUT, "explainability.json"))
    crawl = read_json(os.path.join(ROOT, "data", "crawl", "manifest.json"), {})
    return ana, met, exp, crawl


# ---------------------------------------------------------------- 图1 架构
def fig1_architecture(ana, met, crawl):
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    layers = [
        ("数据源与合规采集", "#eaf2fb", [
            "ScienceDB 公开图文数据集\n64,347 条原始记录 · CC BY 4.0",
            f"清洗去重后全量\n{ana['data']['total_rows']:,} 条有效帖",
            f"1,000 条落盘样本\n匿名 user_id · 时间 · 文本 · 图片",
            "通用合规采集器\nrobots 检查 · ≥2 秒/请求",
        ]),
        ("预处理与特征工程", "#eaf7ec", [
            "文本清洗 / 去重 / 匿名化\n原始微博 ID → post_hash",
            "字符 TF-IDF（2~4 gram）\n→ SVD 256 维 → 标准化",
            "ResNet18 冻结 512 维\n+ 7 维图像质量特征",
            f"建模样本 {ana['data']['sample_rows']:,} 条\n三类各 4,000 条",
        ]),
        ("模型、缺失评估与解释", "#fdf6e3", [
            "单模态基线\n文本 LR / 图像 LR",
            "融合对照\n早期融合 / 晚期融合 / 早期 MLP",
            "门控融合（本文）\n模态 dropout + 缺失嵌入",
            "缺失场景 + 外部无图\n门控权重 · 文本/图像遮挡",
        ]),
        ("应用与展示", "#fbeaea", [
            "舆情分析\n趋势 · 话题 · 关键词 · 省份",
            "异常预警\n负向率/声量 z-score",
            "离线大屏\n14 个视图 · 双击打开",
            "研究报告 / 开发文档\n可复现流水线",
        ]),
    ]
    h, gap = 1.18, 0.30
    y = 6.2
    centers = []
    for title, bg, items in layers:
        y -= h
        ax.add_patch(FancyBboxPatch((0.55, y), 10.1, h, boxstyle="round,pad=0.02",
                                    fc=bg, ec="#9db6d0", lw=1.1))
        ax.text(0.75, y + h / 2, title, ha="left", va="center", fontsize=9.5,
                fontweight="bold", color="#1a3a5c", rotation=0)
        xs = [3.65, 5.55, 7.45, 9.35]
        for x, it in zip(xs, items):
            ax.text(x, y + h / 2, it, ha="center", va="center", fontsize=7.4,
                    bbox=dict(boxstyle="round,pad=0.26", fc="white", ec="#b9c9db", lw=.8))
        centers.append(y + h)
        y -= gap
    for i in range(len(layers) - 1):
        y0 = centers[i] - 0.02
        ax.add_patch(FancyArrowPatch((5.6, y0), (5.6, y0 - gap - 0.02),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color="#5b7ea6", lw=1.4))
    ax.set_title("图1  系统总体架构：公开数据 → 多模态特征 → 门控融合 → 舆情应用",
                 fontsize=11.5, pad=6)
    save(fig, "fig1_architecture.png")


# ---------------------------------------------------------------- 图2 数据概览
def fig2_data_overview(ana, crawl):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.2))
    labs = [-1, 0, 1]
    names = [LABEL_NAMES[x] for x in labs]
    colors = [C_NEG, C_NEU, C_POS]

    ax = axes[0, 0]
    counts = [ana["data"]["label_counts"][str(x)] for x in labs]
    bars = ax.bar(names, counts, color=colors, width=.55)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.015,
                f"{v:,}\n({v / sum(counts):.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_ylabel("帖子数")
    ax.set_title("(a) 全量数据情感分布")

    ax = axes[0, 1]
    dated = ana["data"]["date_rows"]
    undated = ana["data"]["total_rows"] - dated
    bars = ax.bar(["有日期", "无日期"], [dated, undated], color=[C_MAIN, "#c7d2e3"], width=.5)
    for b, v in zip(bars, [dated, undated]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.015,
                f"{v:,}\n({v / ana['data']['total_rows']:.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, undated * 1.18)
    ax.set_ylabel("帖子数")
    ax.set_title("(b) 日期回填覆盖（精确文本匹配）")

    ax = axes[1, 0]
    sample = pd.read_csv(os.path.join(PROCESSED, "posts_clean.csv"), dtype=str)
    sample["text_len"] = pd.to_numeric(sample["text_len"], errors="coerce").fillna(0).clip(0, 200)
    sample["label"] = sample["label"].astype(int)
    bins = np.linspace(0, 200, 41)
    for lab, c, n in zip(labs, colors, names):
        sub = sample[sample["label"] == lab]["text_len"]
        ax.hist(sub, bins=bins, alpha=.48, color=c, label=f"{n}（均值 {sub.mean():.0f}）")
    ax.set_xlabel("清洗后文本长度（字，截断至 200）")
    ax.set_ylabel("帖子数")
    ax.axvline(140, color="#888", ls="--", lw=.8)
    ax.text(142, ax.get_ylim()[1] * .92, "平台 140 字上限", fontsize=7, color="#666")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("(c) 建模样本文本长度分布")

    ax = axes[1, 1]
    cl = crawl.get("label_counts", {})
    vals = [cl.get("-1", cl.get(-1, 0)), cl.get("0", cl.get(0, 0)), cl.get("1", cl.get(1, 0))]
    bars = ax.bar(names, vals, color=colors, width=.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(vals) * 1.30)
    ax.set_ylabel("帖子数")
    ax.set_title("(d) 1,000 条落盘样本（匿名 user_id + 时间 + 图文）")
    ax.text(0.5, 0.86,
            f"{crawl.get('n', 0):,} 条 · {crawl.get('unique_user_id', 0):,} 个 user_id · "
            f"{crawl.get('images', 0):,} 张图\n"
            f"{crawl.get('date_min', '')} ~ {crawl.get('date_max', '')}",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f8fc", ec="#c7d2e3"))
    fig.suptitle("图2  数据概览：全量、建模样本与 1,000 条落盘样本", fontsize=11.5, y=1.01)
    fig.tight_layout()
    save(fig, "fig2_data_overview.png")


# ---------------------------------------------------------------- 图3 日度趋势
def fig3_daily(ana):
    daily = pd.DataFrame([d for d in ana["daily"] if d["n"] >= 50])
    daily["date"] = pd.to_datetime(daily["date"])
    fig, ax = plt.subplots(figsize=(10.6, 4.0))
    ax.bar(daily["date"], daily["n"], color="#9fc3e8", width=.72, label="日声量（条）")
    ax.set_ylabel("日声量（条）")
    ax.set_ylim(0, daily["n"].max() * 1.20)
    ax.axvline(pd.Timestamp("2020-01-20"), color=C_ALT, ls="--", lw=1.2)
    ax.text(pd.Timestamp("2020-01-20"), daily["n"].max() * 1.08, " 1月20日",
            color=C_ALT, fontsize=8, ha="left")
    ax2 = ax.twinx()
    ax2.plot(daily["date"], daily["neg_ratio"] * 100, color=C_NEG, lw=1.6, label="日负向率")
    ax2.plot(daily["date"], daily["neg_ratio_7d"] * 100, color=C_PURPLE, lw=1.2, ls="--",
             label="7日负向率")
    ax2.set_ylabel("负向率（%）")
    ax2.set_ylim(0, max(30, daily["neg_ratio"].max() * 110))
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, frameon=False, ncol=3)
    ax.set_title(f"图3  日度声量与负向率（{daily['date'].min():%Y-%m-%d} ~ {daily['date'].max():%Y-%m-%d}，n≥50）")
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    save(fig, "fig3_daily_trend.png")


# ---------------------------------------------------------------- 图4 话题
def fig4_topics(ana):
    topics = ana["topics"]["list"]
    d = pd.DataFrame(topics).sort_values("n", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    colors = plt.cm.Blues(np.linspace(.38, .82, len(d)))
    bars = ax.barh(d["label"], d["n"], color=colors)
    for b, v, s in zip(bars, d["n"], d["share"]):
        ax.text(v + d["n"].max() * .012, b.get_y() + b.get_height() / 2,
                f"{v:,}（{s:.1%}）", va="center", fontsize=7.6)
    ax.set_xlabel("帖子数")
    ax.set_xlim(0, d["n"].max() * 1.22)
    ax.set_title("(a) 话题声量（NMF k=8）")

    ax = axes[1]
    bars = ax.barh(d["label"], d["neg_ratio"] * 100, color=C_NEG, alpha=.75)
    for b, v in zip(bars, d["neg_ratio"] * 100):
        ax.text(v + .25, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontsize=7.6)
    ax.axvline(ana["trend"]["overall_neg_ratio"] * 100, color=C_ALT, ls="--", lw=1.1)
    ax.text(ana["trend"]["overall_neg_ratio"] * 100 + .2, len(d) - .35,
            f"全量 {ana['trend']['overall_neg_ratio']:.1%}", color=C_ALT, fontsize=7.6)
    ax.set_xlabel("负向率（%）")
    ax.set_xlim(0, max(d["neg_ratio"].max() * 125, 5))
    ax.set_title("(b) 各话题负向率")
    fig.suptitle("图4  话题结构与负向率（TF-IDF + NMF，k=8）", fontsize=11.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_topics.png")


# ---------------------------------------------------------------- 图5 关键词
def fig5_keywords(ana):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, lab, color, title in [
        (axes[0], "-1", C_NEG, "(a) 负向高辨识度词"),
        (axes[1], "1", C_POS, "(b) 正向高辨识度词"),
    ]:
        rows = ana["keywords"][lab][:10]
        d = pd.DataFrame(rows).iloc[::-1]
        bars = ax.barh(d["word"], d["score"], color=color, alpha=.82)
        for b, v, c in zip(bars, d["score"], d["count"]):
            ax.text(v + d["score"].max() * .012, b.get_y() + b.get_height() / 2,
                    f"{v:.2f}（{c}）", va="center", fontsize=7.4)
        ax.set_xlabel("对数几率比 × log(1+词频)")
        ax.set_xlim(0, d["score"].max() * 1.25)
        ax.set_title(title)
    fig.suptitle("图5  情感关键词（排除人名/地名/机构名/专名）", fontsize=11.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig5_keywords.png")


# ---------------------------------------------------------------- 图6 模型对比
def fig6_model_comparison(met):
    order = ["text_lr", "image_lr", "early_lr", "late_lr", "gated", "gated_equal", "early_mlp"]
    labels = {
        "text_lr": "文本单模态", "image_lr": "图像单模态", "early_lr": "早期融合 LR",
        "late_lr": "晚期融合", "gated": "门控融合（本文）", "gated_equal": "门控-等权消融",
        "early_mlp": "早期融合 MLP",
    }
    rows = [met["complete"][k] for k in order]
    vals = [r["macro_f1"] for r in rows]
    lows = [r["macro_f1"] - r["bootstrap_macro_f1"]["low"] for r in rows]
    highs = [r["bootstrap_macro_f1"]["high"] - r["macro_f1"] for r in rows]
    colors = [C_MAIN if k != "gated" else C_GATED for k in order]
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    x = np.arange(len(order))
    bars = ax.bar(x, vals, color=colors, width=.58,
                  yerr=[lows, highs], capsize=3, error_kw={"lw": 1, "ecolor": "#555"})
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(met["complete"]["text_lr"]["macro_f1"], color=C_ALT, ls="--", lw=1.1,
               label="文本单模态")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in order], rotation=18, ha="right")
    ax.set_ylim(0, .80)
    ax.set_ylabel("Macro-F1")
    ax.set_title("图6  完整模态模型对比（测试集 1,800 条，误差线为 bootstrap 95% CI）")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig6_model_comparison.png")


# ---------------------------------------------------------------- 图7 缺失鲁棒
def fig7_robustness(met):
    scen = [("text-only", "仅文本"), ("image-only", "仅图像"), ("both-missing", "双缺失")]
    models = [("early_lr", "早期融合 LR", "#9ecae1"), ("late_lr", "晚期融合", "#6baed6"),
              ("gated", "门控融合（本文）", C_GATED), ("early_mlp", "早期融合 MLP", C_ALT)]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    width = .19
    x = np.arange(len(scen))
    for i, (key, label, color) in enumerate(models):
        vals = []
        for sc, _ in scen:
            row = next(r for r in met["robustness"] if r["model"] == key and r["scenario"] == sc)
            vals.append(row["macro_f1"])
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)
        if key == "gated":
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + .012, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.6, color="#1b5e20")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in scen])
    ax.set_ylim(0, .78)
    ax.set_ylabel("Macro-F1")
    ax.set_title("图7  模态缺失场景下的鲁棒性（测试集 1,800 条）")
    ax.legend(fontsize=7.6, frameon=False, ncol=4, loc="upper right")
    ax.text(.01, .97, "双缺失场景退化为类别先验，仅表示能力下限。",
            transform=ax.transAxes, fontsize=7.2, color="#666", va="top")
    fig.tight_layout()
    save(fig, "fig7_missing_robustness.png")


# ---------------------------------------------------------------- 图8 混淆矩阵
def fig8_confusion(met):
    m = met["complete"]["gated"]
    cm = np.array(m["confusion_matrix"], dtype=float)
    row = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(cm[i, j])}\n({row[i, j]:.1%})", ha="center", va="center",
                    fontsize=9, color="white" if row[i, j] > .55 else "#1a3a5c")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([LABEL_NAMES[x] for x in [-1, 0, 1]])
    ax.set_yticklabels([LABEL_NAMES[x] for x in [-1, 0, 1]])
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"图8  门控融合混淆矩阵（Acc={m['accuracy']:.3f}, Macro-F1={m['macro_f1']:.3f}）")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="行归一化比例")
    fig.tight_layout()
    save(fig, "fig8_confusion.png")


# ---------------------------------------------------------------- 图9 可解释性
def fig9_explainability(met, exp):
    fig = plt.figure(figsize=(12.4, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 1.0], wspace=.34)

    ax = fig.add_subplot(gs[0, 0])
    gates = [g for g in met["gates"] if g["group"] == "true_label"]
    labels = [g["label"] for g in gates]
    t = np.array([g["text_gate"] for g in gates]) * 100
    i = np.array([g["image_gate"] for g in gates]) * 100
    y = np.arange(len(labels))
    ax.barh(y, t, color=C_MAIN, label="文本")
    ax.barh(y, i, left=t, color=C_GATED, label="图像")
    for yy, tv, iv in zip(y, t, i):
        ax.text(tv / 2, yy, f"{tv:.1f}%", ha="center", va="center", color="white", fontsize=7.6)
        if iv > 2:
            ax.text(tv + iv / 2, yy, f"{iv:.1f}%", ha="center", va="center",
                    color="white", fontsize=7.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("门控权重（%）")
    ax.set_title("(a) 门控权重（蓝=文本，绿=图像）")
    ax.grid(axis="x", alpha=.25)

    ax = fig.add_subplot(gs[0, 1])
    toks = exp["text_occlusion"]["top_tokens"][:8][::-1]
    d = pd.DataFrame(toks)
    bars = ax.barh(d["token"], d["mean_prob_drop"], color=C_NEG, alpha=.8)
    for b, v in zip(bars, d["mean_prob_drop"]):
        ax.text(v + d["mean_prob_drop"].max() * .02, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=7.2)
    ax.set_xlabel("真实类概率平均下降")
    ax.set_xlim(0, d["mean_prob_drop"].max() * 1.22)
    ax.set_title("(b) 文本 token 遮挡")

    ax = fig.add_subplot(gs[0, 2])
    grid = np.array(exp["image_occlusion"]["grid"])
    im = ax.imshow(grid, cmap="Oranges", vmin=0, vmax=max(grid.max(), 1e-6))
    for r in range(3):
        for c in range(3):
            ax.text(c, r, f"{grid[r, c]:.4f}", ha="center", va="center",
                    fontsize=8, color="#3a1a00" if grid[r, c] < grid.max() * .7 else "white")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["左", "中", "右"])
    ax.set_yticklabels(["上", "中", "下"])
    ax.set_title("(c) 图像 3×3 遮挡（仅图像）")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.suptitle("图9  可解释性：门控权重、文本 token 遮挡、图像区域遮挡", fontsize=11.5, y=1.03)
    save(fig, "fig9_explainability.png")


# ---------------------------------------------------------------- 图10 外部测试
def fig10_external(met):
    order = ["text_lr", "early_lr", "late_lr", "gated", "early_mlp"]
    labels = {"text_lr": "文本单模态", "early_lr": "早期融合 LR", "late_lr": "晚期融合",
              "gated": "门控融合（本文）", "early_mlp": "早期融合 MLP"}
    rows = [met["external_no_image"][k] for k in order]
    vals = [r["macro_f1"] for r in rows]
    lows = [r["macro_f1"] - r["bootstrap_macro_f1"]["low"] for r in rows]
    highs = [r["bootstrap_macro_f1"]["high"] - r["macro_f1"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    x = np.arange(len(order))
    colors = [C_MAIN if k != "gated" else C_GATED for k in order]
    bars = ax.bar(x, vals, color=colors, width=.56, yerr=[lows, highs], capsize=3,
                  error_kw={"lw": 1, "ecolor": "#555"})
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in order], rotation=16, ha="right")
    ax.set_ylim(0, .72)
    ax.set_ylabel("Macro-F1")
    ax.set_title("图10  真实无图外部测试（3,000 条，误差线为 bootstrap 95% CI）")
    fig.tight_layout()
    save(fig, "fig10_external.png")


EDGE_CANDIDATES = [
    os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                 "Microsoft", "Edge", "Application", "msedge.exe"),
    os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                 "Microsoft", "Edge", "Application", "msedge.exe"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""),
                 "Microsoft", "Edge", "Application", "msedge.exe"),
]


def find_edge():
    for p in EDGE_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return shutil.which("msedge") or shutil.which("microsoft-edge")


def screenshot_dashboard():
    """用 Edge 无头模式截取 1920×1080 大屏；失败时保留旧图，不阻断流水线。"""
    edge = find_edge()
    out = os.path.join(FIGURES, "dashboard.png")
    if not edge:
        print("  [WARN] 未找到 Edge，跳过大屏截图（保留旧图）")
        return
    html = os.path.join(ROOT, "dashboard", "index.html")
    if not os.path.exists(html):
        print("  [WARN] 缺少 dashboard/index.html，跳过大屏截图")
        return
    profile = tempfile.mkdtemp(prefix="edge_shot_")
    try:
        cmd = [
            edge, "--headless", "--disable-gpu", "--hide-scrollbars",
            "--no-first-run", "--force-device-scale-factor=1",
            "--window-size=1920,1080", "--virtual-time-budget=8000",
            "--user-data-dir=" + profile,
            "--screenshot=" + out,
            "file:///" + html.replace("\\", "/"),
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)
        if os.path.exists(out) and os.path.getsize(out) > 50_000:
            print(f"  -> {os.path.relpath(out, ROOT)}（大屏截图）")
        else:
            print("  [WARN] 大屏截图失败（Edge 未写出有效文件），保留旧图")
    except Exception as e:
        print(f"  [WARN] 大屏截图异常：{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def main():
    ana, met, exp, crawl = load()
    print("06 生成报告插图")
    fig1_architecture(ana, met, crawl)
    fig2_data_overview(ana, crawl)
    fig3_daily(ana)
    fig4_topics(ana)
    fig5_keywords(ana)
    fig6_model_comparison(met)
    fig7_robustness(met)
    fig8_confusion(met)
    fig9_explainability(met, exp)
    fig10_external(met)
    screenshot_dashboard()
    print("[完成] 10 张图 + 大屏截图已生成。")


if __name__ == "__main__":
    main()
