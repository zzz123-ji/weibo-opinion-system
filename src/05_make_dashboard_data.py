# -*- coding: utf-8 -*-
"""05 汇总大屏数据：dashboard/data.js（window.DASH_DATA，离线双击可开）。

输入
----
output/analysis_results.json
output/models_metrics.json
output/explainability.json
data/crawl/manifest.json

输出
----
dashboard/data.js
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import (DASHBOARD, DATASET_DOI, DATASET_LICENSE, OUT, ROOT,
                        now_str, read_json, write_json)


def main():
    ana = read_json(os.path.join(OUT, "analysis_results.json"))
    met = read_json(os.path.join(OUT, "models_metrics.json"))
    exp = read_json(os.path.join(OUT, "explainability.json"), {})
    crawl = read_json(os.path.join(ROOT, "data", "crawl", "manifest.json"), {})
    if not ana or not met:
        raise SystemExit("缺少 analysis_results.json 或 models_metrics.json，请先运行 03/04")

    # 日度趋势：过滤极低样本日（n<50），避免单条样本造成假波动
    daily = [d for d in ana["daily"] if int(d["n"]) >= 50]
    sentiment = []
    for lab in ["-1", "0", "1"]:
        sentiment.append({
            "name": {"-1": "负面", "0": "中性", "1": "正面"}[lab],
            "value": round(ana["data"]["label_ratio"][lab] * 100, 2),
            "count": ana["data"]["label_counts"][lab],
        })

    topics = [{
        "label": t["label"],
        "n": t["n"],
        "share": round(t["share"] * 100, 2),
        "neg_ratio": round(t["neg_ratio"] * 100, 2),
        "pos_ratio": round(t["pos_ratio"] * 100, 2),
        "terms": t["top_terms"][:4],
    } for t in ana["topics"]["list"]]

    def kw(lab, n=10):
        return [{"word": x["word"], "count": x["count"], "score": x["score"]}
                for x in ana["keywords"][lab][:n]]

    complete = []
    short_names = {
        "text_lr": "文本", "image_lr": "图像", "early_lr": "早期LR",
        "late_lr": "晚期", "gated": "门控*", "gated_equal": "等权",
        "early_mlp": "早期MLP",
    }
    for name in ["text_lr", "image_lr", "early_lr", "late_lr", "gated", "gated_equal", "early_mlp"]:
        m = met["complete"].get(name)
        if not m:
            continue
        complete.append({
            "model": name,
            "label": {
                "text_lr": "文本单模态", "image_lr": "图像单模态",
                "early_lr": "早期融合LR", "late_lr": "晚期融合",
                "gated": "门控融合(本文)", "gated_equal": "门控-等权",
                "early_mlp": "早期融合MLP",
            }[name],
            "short": short_names[name],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "ci_low": m["bootstrap_macro_f1"]["low"],
            "ci_high": m["bootstrap_macro_f1"]["high"],
        })

    robustness = []
    for r in met["robustness"]:
        robustness.append({
            "model": r["model"],
            "label": {
                "early_lr": "早期融合LR", "late_lr": "晚期融合",
                "gated": "门控融合(本文)", "early_mlp": "早期融合MLP",
            }.get(r["model"], r["model"]),
            "scenario": r["scenario"],
            "accuracy": r["accuracy"],
            "macro_f1": r["macro_f1"],
        })

    external = []
    ext_short = {"text_lr": "文本", "early_lr": "早期LR", "late_lr": "晚期",
                 "gated": "门控*", "early_mlp": "早期MLP"}
    for name in ["text_lr", "early_lr", "late_lr", "gated", "early_mlp"]:
        m = met["external_no_image"].get(name)
        if not m:
            continue
        external.append({
            "model": name,
            "label": {
                "text_lr": "文本单模态", "early_lr": "早期融合LR",
                "late_lr": "晚期融合", "gated": "门控融合(本文)",
                "early_mlp": "早期融合MLP",
            }[name],
            "short": ext_short[name],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "ci_low": m["bootstrap_macro_f1"]["low"],
            "ci_high": m["bootstrap_macro_f1"]["high"],
        })

    gates = [g for g in met.get("gates", []) if g["group"] == "true_label"]
    alerts = ana.get("alerts", [])[:8]
    text_occ = (exp or {}).get("text_occlusion", {})
    image_occ = (exp or {}).get("image_occlusion", {})

    data = {
        "meta": {
            "generated_at": now_str(),
            "title": "面向社交媒体舆情的模态缺失鲁棒、可解释多模态情感分析系统",
            "subtitle": "以微博为例 · 2020 年疫情早期公开图文帖 · 中国大学生计算机设计大赛 大数据应用-大数据实践赛",
            "source": "ScienceDB《Weibo and Twitter datasets》",
            "doi": DATASET_DOI,
            "license": DATASET_LICENSE,
            "date_min": ana["data"].get("date_min"),
            "date_max": ana["data"].get("date_max"),
        },
        "kpi": {
            "total": ana["data"]["total_rows"],
            "sample": ana["data"]["sample_rows"],
            "crawl": int(crawl.get("n", 0)),
            "crawl_users": int(crawl.get("unique_user_id", 0)),
            "neg_pct": round(ana["data"]["label_ratio"]["-1"] * 100, 2),
            "volume_growth": round(ana["trend"]["volume_growth"] * 100, 1),
            "gated_f1": met["complete"]["gated"]["macro_f1"],
            "external_f1": met["external_no_image"]["gated"]["macro_f1"],
            "alerts": len(ana.get("alerts", [])),
            "vocab": ana["data"].get("vocab_size", 0),
        },
        "daily": [{
            "date": d["date"],
            "n": d["n"],
            "neg_ratio": round(d["neg_ratio"] * 100, 2),
            "pos_ratio": round(d["pos_ratio"] * 100, 2),
            "mean_sentiment": d["mean_sentiment"],
            "neg_ratio_7d": round((d.get("neg_ratio_7d") or 0) * 100, 2),
            "low_sample": d.get("low_sample", False),
        } for d in daily],
        "trend": ana["trend"],
        "sentiment": sentiment,
        "topics": topics,
        "keywords": {"neg": kw("-1"), "pos": kw("1"), "neu": kw("0")},
        "provinces": [{
            "province": p["province"], "n": p["n"],
            "neg_ratio": round(p["neg_ratio"] * 100, 2),
            "pos_ratio": round(p["pos_ratio"] * 100, 2),
        } for p in ana["provinces"][:10]],
        "hashtags": [{
            "tag": h["tag"], "n": h["n"],
            "neg_ratio": round(h["neg_ratio"] * 100, 2),
            "pos_ratio": round(h["pos_ratio"] * 100, 2),
        } for h in ana["hashtags"][:10]],
        "alerts": alerts,
        "models": {
            "complete": complete,
            "robustness": robustness,
            "external": external,
        },
        "gates": gates,
        "explainability": {
            "text_occlusion": {
                "n_samples": text_occ.get("n_samples", 0),
                "top_tokens": text_occ.get("top_tokens", [])[:10],
            },
            "image_occlusion": {
                "n_samples": image_occ.get("n_samples", 0),
                "grid": image_occ.get("grid", []),
            },
        },
        "crawl": {
            "n": int(crawl.get("n", 0)),
            "unique_user_id": int(crawl.get("unique_user_id", 0)),
            "images": int(crawl.get("images", 0)),
            "image_total_mb": crawl.get("image_total_mb", 0),
            "date_min": crawl.get("date_min", ""),
            "date_max": crawl.get("date_max", ""),
            "label_counts": crawl.get("label_counts", {}),
        },
    }
    path = os.path.join(DASHBOARD, "data.js")
    with open(path, "w", encoding="utf-8") as f:
        f.write("window.DASH_DATA = ")
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    print(f"大屏数据 -> {path}")
    print(f"  日度 {len(daily)} 天；话题 {len(topics)}；模型 {len(complete)}；"
          f"采集 {data['crawl']['n']:,} 条")


if __name__ == "__main__":
    main()
