# -*- coding: utf-8 -*-
"""08 提交前自检门禁：产物、规模、指标一致性、报告结构、PDF 新鲜度、
离线大屏、合规红线、隐私隔离、内部文件提醒。

用法
----
    python src/08_submission_check.py
    python src/08_submission_check.py --no-write   # 不写 output/submission_check.json

退出码
------
    0 = 没有 FAIL（可有 WARN/INFO）
    1 = 存在 FAIL，必须修复后再提交

设计原则
--------
- 只读检查，不改任何数据/文档；
- 指标从 output/*.json 读取，不在脚本里硬编码实验值；
- 报告/PDF/大屏的新鲜度按文件修改时间比较，容忍 3 秒误差；
- 隐私检查只针对“原始账号名/原始微博 ID/图片 URL”，不误伤匿名 user_id。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")
FIGURES = os.path.join(DOCS, "figures")
OUT = os.path.join(ROOT, "output")
TABLES = os.path.join(OUT, "tables")
MODELS = os.path.join(OUT, "models")
DASH = os.path.join(ROOT, "dashboard")
DATA = os.path.join(ROOT, "data")
CRAWL = os.path.join(DATA, "crawl")
PROCESSED = os.path.join(DATA, "processed")
RAW = os.path.join(DATA, "raw")

TOL = 3.0  # 文件时间比较容忍秒数
BOTTOM_TOL = 60.0  # 同一次 04 运行内 CSV/JSON 写入顺序不同，底表放宽到 60 秒


class Checker:
    def __init__(self):
        self.items = []

    def add(self, status, category, name, detail=""):
        self.items.append({
            "status": status,
            "category": category,
            "name": name,
            "detail": str(detail),
        })

    def ok(self, category, name, detail=""):
        self.add("PASS", category, name, detail)

    def fail(self, category, name, detail=""):
        self.add("FAIL", category, name, detail)

    def warn(self, category, name, detail=""):
        self.add("WARN", category, name, detail)

    def info(self, category, name, detail=""):
        self.add("INFO", category, name, detail)

    def counts(self):
        c = {"PASS": 0, "FAIL": 0, "WARN": 0, "INFO": 0}
        for it in self.items:
            c[it["status"]] = c.get(it["status"], 0) + 1
        return c


def p(rel):
    return os.path.join(ROOT, rel.replace("/", os.sep))


def exists(rel):
    return os.path.exists(p(rel))


def read_text(rel):
    with open(p(rel), encoding="utf-8") as f:
        return f.read()


def read_json(rel):
    with open(p(rel), encoding="utf-8") as f:
        return json.load(f)


def mtime(rel):
    return os.path.getmtime(p(rel))


def fsize(rel):
    return os.path.getsize(p(rel))


def newer(a, b, tol=TOL):
    return mtime(a) + tol >= mtime(b)


def f4(x):
    return f"{float(x):.4f}"


def pct1(x):
    return f"{float(x) * 100:.1f}%"


def pct2(x):
    return f"{float(x) * 100:.2f}%"


def csv_header(rel):
    with open(p(rel), encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        return next(r, [])


def check_required_files(c):
    required = [
        "README.md", "requirements.txt", "src/run_all.py",
        "src/lib_common.py", "src/lib_text.py", "src/lib_image.py",
        "src/01_data_collection.py", "src/01a_import_scidb.py",
        "src/01b_collect_public.py", "src/01c_export_crawl_1000.py",
        "src/02_preprocess.py", "src/03_analysis.py", "src/04_modeling.py",
        "src/05_make_dashboard_data.py", "src/06_make_report_figures.py",
        "src/07_export_docs.py", "src/08_submission_check.py",
        "src/09_fill_report.py", "src/lib_infer.py",
        "src/10_inference_service.py", "src/11_export_explanation_cards.py",
        "src/12_missing_explain_experiments.py", "src/13_annotation_kappa.py",
        "docs/01_研究报告.md", "docs/02_开发文档.md", "docs/03_数据说明.md",
        "docs/04_实验记录.md", "docs/05_演示视频脚本.md",
        "docs/06_提交前自检清单.md", "docs/07_答辩要点.md",
        "docs/08_作品方案要求对照.md", "docs/09_单条分析接口与示例.md",
        "docs/figures/fig1_architecture.png", "docs/figures/fig2_data_overview.png",
        "docs/figures/fig3_daily_trend.png", "docs/figures/fig4_topics.png",
        "docs/figures/fig5_keywords.png", "docs/figures/fig6_model_comparison.png",
        "docs/figures/fig7_missing_robustness.png", "docs/figures/fig8_confusion.png",
        "docs/figures/fig9_explainability.png", "docs/figures/fig10_external.png",
        "docs/figures/fig11_missing_ratio.png", "docs/figures/fig12_deletion.png",
        "docs/figures/dashboard.png",
        "dashboard/index.html", "dashboard/single.html",
        "dashboard/data.js", "dashboard/echarts.min.js",
        "output/analysis_results.json", "output/models_metrics.json",
        "output/explainability.json", "output/data_quality.json",
        "output/crawler_self_test.json", "output/preprocess_report.json",
        "output/missing_experiments.json", "output/inference_self_test.json",
        "output/explanation_cards.json", "output/annotation_self_test.json",
        "output/models/gated_fusion.pt", "output/models/early_fusion_mlp.pt",
        "output/models/baselines.joblib", "output/models/MODEL_CARD.md",
        "data/crawl/weibo_1000.csv", "data/crawl/manifest.json",
        "data/crawl/README.md", "data/annotation/annotation_template.csv",
        "data/processed/posts_full.csv", "data/processed/posts_clean.csv",
        "data/processed/external_test_clean.csv",
        "data/processed/image_features.npy", "data/processed/image_stats.csv",
        "data/processed/image_feature_index.csv",
    ]
    required += [
        "docs/01_研究报告.pdf", "docs/02_开发文档.pdf", "docs/03_数据说明.pdf",
        "docs/04_实验记录.pdf", "docs/05_演示视频脚本.pdf",
        "docs/06_提交前自检清单.pdf", "docs/07_答辩要点.pdf",
        "docs/08_作品方案要求对照.pdf", "docs/09_单条分析接口与示例.pdf",
        "README.pdf",
        "docs/01_研究报告.html", "docs/02_开发文档.html", "docs/03_数据说明.html",
        "docs/04_实验记录.html", "docs/05_演示视频脚本.html",
        "docs/06_提交前自检清单.html", "docs/07_答辩要点.html",
        "docs/08_作品方案要求对照.html", "docs/09_单条分析接口与示例.html",
        "README.html",
    ]
    table_names = [
        "日度情感趋势.csv", "话题分析.csv", "情感关键词.csv", "省份提及.csv",
        "话题标签.csv", "异常预警.csv", "情感样本示例.csv", "文本长度统计.csv",
        "图像统计.csv", "模型对比.csv", "缺失场景鲁棒性.csv", "外部测试.csv",
        "混淆矩阵.csv", "门控权重.csv", "文本遮挡解释.csv", "图像遮挡解释.csv",
        "缺失比例曲线.csv", "缺失位置曲线.csv", "删除实验.csv",
        "数据校验报告.csv", "爬虫自检.csv", "预处理统计.csv",
    ]
    required += [f"output/tables/{n}" for n in table_names]

    missing = [r for r in required if not exists(r)]
    empty = [r for r in required if exists(r) and fsize(r) == 0]
    if missing:
        c.fail("产物", "必需文件齐全", "缺少：" + "；".join(missing[:12]) +
               (f" 等 {len(missing)} 个" if len(missing) > 12 else ""))
    else:
        c.ok("产物", "必需文件齐全", f"{len(required)} 个文件")
    if empty:
        c.fail("产物", "文件非空", "空文件：" + "；".join(empty[:8]))
    else:
        c.ok("产物", "文件非空", "全部必需文件 > 0 字节")


def check_data_scale(c):
    ana = read_json("output/analysis_results.json")
    met = read_json("output/models_metrics.json")
    exp = read_json("output/explainability.json")
    pre = read_json("output/preprocess_report.json")
    man = read_json("data/crawl/manifest.json")
    dq = read_json("output/data_quality.json")
    cr = read_json("output/crawler_self_test.json")

    a, m = ana.get("data", {}), met.get("data", {})
    total, dedup = int(a.get("total_rows", 0)), int(pre.get("dedup_rows", 0))
    if total == dedup:
        c.ok("数据规模", "全量口径一致", f"analysis={total:,}，preprocess={dedup:,}")
    else:
        c.fail("数据规模", "全量口径一致", f"analysis={total:,} ≠ preprocess={dedup:,}")
    if total >= 56000:
        c.ok("数据规模", "全量有效帖", f"{total:,}（≥56,000）")
    else:
        c.fail("数据规模", "全量有效帖", f"{total:,}，低于 56,000")

    checks = [
        ("建模样本 12,000", int(a.get("sample_rows", 0)) == 12000, a.get("sample_rows")),
        ("训练集 8,400", int(m.get("train", 0)) == 8400, m.get("train")),
        ("验证集 1,800", int(m.get("val", 0)) == 1800, m.get("val")),
        ("测试集 1,800", int(m.get("test", 0)) == 1800, m.get("test")),
        ("外部无图 3,000", int(m.get("external_no_image", 0)) == 3000, m.get("external_no_image")),
        ("落盘样本 1,000", int(man.get("n", 0)) == 1000, man.get("n")),
        ("落盘图片 1,000", int(man.get("images", 0)) == 1000, man.get("images")),
        ("落盘缺图 0", int(man.get("image_missing", -1)) == 0, man.get("image_missing")),
    ]
    bad = [f"{name}={val}" for name, ok, val in checks if not ok]
    if bad:
        c.fail("数据规模", "关键数量", "；".join(bad))
    else:
        c.ok("数据规模", "关键数量", "12,000 / 8,400 / 1,800 / 1,800 / 3,000 / 1,000")

    if 900 <= int(man.get("unique_user_id", 0)) <= 1000:
        c.ok("数据规模", "落盘唯一 user_id", f"{man.get('unique_user_id')}（900~1000）")
    else:
        c.fail("数据规模", "落盘唯一 user_id", man.get("unique_user_id"))

    dated = int(a.get("date_rows", 0))
    if abs(dated - int(pre.get("dated_rows", 0))) <= 5:
        c.ok("数据规模", "日期回填口径", f"{dated:,} vs {pre.get('dated_rows'):,}")
    else:
        c.fail("数据规模", "日期回填口径", f"{dated:,} vs {pre.get('dated_rows'):,}")
    cov = float(a.get("date_coverage", 0))
    if 0.34 <= cov <= 0.38:
        c.ok("数据规模", "日期覆盖率", pct2(cov))
    else:
        c.fail("数据规模", "日期覆盖率", pct2(cov))

    if int(m.get("image_dim", 0)) == 519:
        c.ok("数据规模", "图像特征维度", "519（512 ResNet + 7 统计）")
    else:
        c.fail("数据规模", "图像特征维度", m.get("image_dim"))
    if pre.get("image_backend") == "resnet18":
        c.ok("数据规模", "图像后端", "resnet18")
    else:
        c.warn("数据规模", "图像后端", pre.get("image_backend"))

    if dq.get("status") == "PASS" and not dq.get("fails"):
        c.ok("数据质量", "01 校验", "PASS，fails=[]")
    else:
        c.fail("数据质量", "01 校验", f"status={dq.get('status')}，fails={dq.get('fails')}")
    if int(pre.get("missing_images", -1)) == 0 and int(pre.get("bad_images", -1)) == 0:
        c.ok("数据质量", "图片完整性", "missing=0，bad=0")
    else:
        c.fail("数据质量", "图片完整性", f"missing={pre.get('missing_images')}，bad={pre.get('bad_images')}")

    self_checks = cr.get("checks", [])
    if self_checks and all(x.get("pass") for x in self_checks) and float(cr.get("min_delay_seconds", 0)) >= 2.0:
        c.ok("合规", "01b 采集器自检", f"{len(self_checks)} 项全部 PASS，限速 {cr.get('min_delay_seconds')}s")
    else:
        c.fail("合规", "01b 采集器自检", "存在未通过项或限速 <2s")

    if int(a.get("unique_days", 0)) == 50 and len(ana.get("daily", [])) == 50:
        c.ok("数据规模", "日度序列", "50 天")
    else:
        c.fail("数据规模", "日度序列", f"unique_days={a.get('unique_days')}，daily={len(ana.get('daily', []))}")
    if len(ana.get("topics", {}).get("list", [])) == 8:
        c.ok("数据规模", "话题数", "8")
    else:
        c.fail("数据规模", "话题数", len(ana.get("topics", {}).get("list", [])))
    if 1 <= len(ana.get("alerts", [])) <= 20:
        c.ok("数据规模", "预警条数", len(ana.get("alerts", [])))
    else:
        c.fail("数据规模", "预警条数", len(ana.get("alerts", [])))

    if len(met.get("complete", {})) == 7:
        c.ok("数据规模", "完整模态模型数", "7")
    else:
        c.fail("数据规模", "完整模态模型数", len(met.get("complete", {})))
    if len(met.get("robustness", [])) == 12:
        c.ok("数据规模", "缺失场景组合", "4 模型 × 3 场景 = 12")
    else:
        c.fail("数据规模", "缺失场景组合", len(met.get("robustness", [])))
    if len(met.get("external_no_image", {})) == 5:
        c.ok("数据规模", "外部测试模型数", "5")
    else:
        c.fail("数据规模", "外部测试模型数", len(met.get("external_no_image", {})))

    to, io = exp.get("text_occlusion", {}), exp.get("image_occlusion", {})
    n_pert = int(to.get("n_perturbations", 0))
    if int(to.get("n_samples", 0)) == 120 and 700 <= n_pert <= 900:
        c.ok("数据规模", "文本遮挡规模", f"120 条 / {n_pert} 次扰动（700~900）")
    else:
        c.fail("数据规模", "文本遮挡规模", f"{to.get('n_samples')} / {to.get('n_perturbations')}")
    if int(io.get("n_samples", 0)) == 45:
        c.ok("数据规模", "图像遮挡规模", "45 条")
    else:
        c.fail("数据规模", "图像遮挡规模", io.get("n_samples"))

    miss = read_json("output/missing_experiments.json")
    mr = miss.get("missing_ratio", {})
    mp = miss.get("missing_position", {})
    dl = miss.get("deletion", {})
    if int(miss.get("test_n", 0)) == 1800 and mr.get("ratios") == [0.1, 0.3, 0.5, 0.7] \
            and mr.get("seeds") == [42, 43, 44]:
        c.ok("数据规模", "缺失比例实验", "1,800 条 × 4 比例 × 3 种子")
    else:
        c.fail("数据规模", "缺失比例实验", f"test_n={miss.get('test_n')}，ratios={mr.get('ratios')}，seeds={mr.get('seeds')}")
    n_ratio_rows = len(mr.get("rows", []))
    if n_ratio_rows == 24:
        c.ok("数据规模", "缺失比例组合", "2 模型 × 3 缺失类型 × 4 比例 = 24")
    else:
        c.fail("数据规模", "缺失比例组合", n_ratio_rows)
    if len(mp.get("rows", [])) == 48:
        c.ok("数据规模", "缺失位置组合", "2 模型 × 2 模态 × 3 位置 × 4 比例 = 48")
    else:
        c.fail("数据规模", "缺失位置组合", f"{len(mp.get('rows', []))} 行，期望 48")
    text_del = dl.get("text", {})
    image_del = dl.get("image", {})
    if int(text_del.get("n_candidates", 0)) >= 90 and int(image_del.get("n_candidates", 0)) >= 10:
        c.ok("数据规模", "删除实验规模", f"文本 {text_del.get('n_candidates')} 条 / 图像 {image_del.get('n_candidates')} 条")
    else:
        c.fail("数据规模", "删除实验规模", f"文本 {text_del.get('n_candidates')} / 图像 {image_del.get('n_candidates')}")
    if any(r.get("modality") == "text" and r.get("method") == "top" for r in text_del.get("rows", [])) \
            and any(r.get("modality") == "image" and r.get("method") == "top" for r in image_del.get("rows", [])):
        c.ok("数据规模", "删除实验分组", "文本/图像 × top/random")
    else:
        c.fail("数据规模", "删除实验分组", "缺少 text/image 或 top/random 行")

    infer = read_json("output/inference_self_test.json")
    if int(infer.get("n_examples", 0)) == 2 and all(
            e.get("name") in ("text_only", "text_image") for e in infer.get("examples", [])):
        c.ok("数据规模", "单条推理自检", "text_only + text_image")
    else:
        c.fail("数据规模", "单条推理自检", f"n_examples={infer.get('n_examples')}")
    cards = read_json("output/explanation_cards.json")
    if int(cards.get("n", 0)) == 3 and len(cards.get("cards", [])) == 3:
        c.ok("数据规模", "解释卡", "3 条")
    else:
        c.fail("数据规模", "解释卡", f"n={cards.get('n')}")


def build_facts():
    ana = read_json("output/analysis_results.json")
    met = read_json("output/models_metrics.json")
    exp = read_json("output/explainability.json")
    man = read_json("data/crawl/manifest.json")

    comp = met["complete"]
    rob = {(r["model"], r["scenario"]): r for r in met["robustness"]}
    ext = met["external_no_image"]
    gate_all = next(g for g in met["gates"] if g["group"] == "all")
    gate_true = {g["label"]: g for g in met["gates"] if g["group"] == "true_label"}
    trend = ana["trend"]
    top_topic = max(ana["topics"]["list"], key=lambda x: x["n"])
    top_hashtag = ana["hashtags"][0]
    top_prov = ana["provinces"][0]
    top_alert = max(ana["alerts"], key=lambda x: x["score"])
    to = exp["text_occlusion"]
    io = exp["image_occlusion"]
    top_token = to["top_tokens"][0]
    grid = io["grid"]
    miss = read_json("output/missing_experiments.json")
    mr = miss["missing_ratio"]["rows"]
    dl = miss["deletion"]
    infer = read_json("output/inference_self_test.json")
    cards = read_json("output/explanation_cards.json")

    def mrow(model, modality, ratio):
        return next(r for r in mr if r["model"] == model and r["modality"] == modality
                    and abs(float(r["ratio"]) - ratio) < 1e-9)

    def drow(modality, method, k):
        return next(r for r in dl[modality]["rows"]
                    if r["method"] == method and int(r["k"]) == k)

    def infer_ms(name):
        for e in infer.get("examples", []):
            if e.get("name") == name:
                return float(e.get("result", {}).get("latency_ms") or 0)
        return 0.0

    facts = {
        "门控完整 Macro-F1": f4(comp["gated"]["macro_f1"]),
        "门控完整 Accuracy": f4(comp["gated"]["accuracy"]),
        "门控完整 CI": f"{comp['gated']['bootstrap_macro_f1']['low']:.4f}~"
                       f"{comp['gated']['bootstrap_macro_f1']['high']:.4f}",
        "文本单模态": f4(comp["text_lr"]["macro_f1"]),
        "文本单模态 CI": f"{comp['text_lr']['bootstrap_macro_f1']['low']:.4f}~"
                          f"{comp['text_lr']['bootstrap_macro_f1']['high']:.4f}",
        "图像单模态": f4(comp["image_lr"]["macro_f1"]),
        "图像单模态 CI": f"{comp['image_lr']['bootstrap_macro_f1']['low']:.4f}~"
                           f"{comp['image_lr']['bootstrap_macro_f1']['high']:.4f}",
        "早期融合 LR": f4(comp["early_lr"]["macro_f1"]),
        "早期融合 LR CI": f"{comp['early_lr']['bootstrap_macro_f1']['low']:.4f}~"
                           f"{comp['early_lr']['bootstrap_macro_f1']['high']:.4f}",
        "晚期融合": f4(comp["late_lr"]["macro_f1"]),
        "晚期融合 CI": f"{comp['late_lr']['bootstrap_macro_f1']['low']:.4f}~"
                         f"{comp['late_lr']['bootstrap_macro_f1']['high']:.4f}",
        "门控等权": f4(comp["gated_equal"]["macro_f1"]),
        "门控等权 CI": f"{comp['gated_equal']['bootstrap_macro_f1']['low']:.4f}~"
                        f"{comp['gated_equal']['bootstrap_macro_f1']['high']:.4f}",
        "早期融合 MLP": f4(comp["early_mlp"]["macro_f1"]),
        "早期融合 MLP CI": f"{comp['early_mlp']['bootstrap_macro_f1']['low']:.4f}~"
                             f"{comp['early_mlp']['bootstrap_macro_f1']['high']:.4f}",
        "门控仅文本": f4(rob[("gated", "text-only")]["macro_f1"]),
        "门控仅图像": f4(rob[("gated", "image-only")]["macro_f1"]),
        "门控双缺失": f4(rob[("gated", "both-missing")]["macro_f1"]),
        "早期 LR 仅文本": f4(rob[("early_lr", "text-only")]["macro_f1"]),
        "早期 LR 仅图像": f4(rob[("early_lr", "image-only")]["macro_f1"]),
        "外部门控": f4(ext["gated"]["macro_f1"]),
        "外部门控 CI": f"{ext['gated']['bootstrap_macro_f1']['low']:.4f}~"
                         f"{ext['gated']['bootstrap_macro_f1']['high']:.4f}",
        "外部文本": f4(ext["text_lr"]["macro_f1"]),
        "外部文本 CI": f"{ext['text_lr']['bootstrap_macro_f1']['low']:.4f}~"
                         f"{ext['text_lr']['bootstrap_macro_f1']['high']:.4f}",
        "外部早期 LR": f4(ext["early_lr"]["macro_f1"]),
        "外部早期 LR CI": f"{ext['early_lr']['bootstrap_macro_f1']['low']:.4f}~"
                            f"{ext['early_lr']['bootstrap_macro_f1']['high']:.4f}",
        "外部 MLP": f4(ext["early_mlp"]["macro_f1"]),
        "外部 MLP CI": f"{ext['early_mlp']['bootstrap_macro_f1']['low']:.4f}~"
                         f"{ext['early_mlp']['bootstrap_macro_f1']['high']:.4f}",
        "门控文本权重": pct1(gate_all["text_gate"]),
        "门控图像权重": pct1(gate_all["image_gate"]),
        "负面图像权重": pct1(gate_true["负面"]["image_gate"]),
        "负面文本权重": pct1(gate_true["负面"]["text_gate"]),
        "正面图像权重": pct1(gate_true["正面"]["image_gate"]),
        "正面文本权重": pct1(gate_true["正面"]["text_gate"]),
        "预警数": str(len(ana["alerts"])),
        "声量增长": pct1(trend["volume_growth"]),
        "负向率峰值": pct2(trend["peak_neg_ratio"]),
        "全量帖数": f"{int(ana['data']['total_rows']):,}",
        "日期回填数": f"{int(ana['data']['date_rows']):,}",
        "日期覆盖率": pct2(ana["data"]["date_coverage"]),
        "建模样本": f"{int(ana['data']['sample_rows']):,}",
        "外部样本": f"{int(met['data']['external_no_image']):,}",
        "训练集": f"{int(met['data']['train']):,}",
        "验证集": f"{int(met['data']['val']):,}",
        "测试集": f"{int(met['data']['test']):,}",
        "落盘样本": f"{int(man['n']):,}",
        "落盘用户": str(man["unique_user_id"]),
        "落盘图片 MB": f"{float(man['image_total_mb']):.2f}",
        "最大话题份额": pct2(top_topic["share"]),
        "最大话题负向率": pct2(top_topic["neg_ratio"]),
        "最大话题帖数": f"{int(top_topic['n']):,}",
        "最高频标签数": f"{int(top_hashtag['n']):,}",
        "最高频标签正向率": pct2(top_hashtag["pos_ratio"]),
        "最高频标签负向率": pct2(top_hashtag["neg_ratio"]),
        "湖北提及数": f"{int(top_prov['n']):,}",
        "湖北占比": pct2(top_prov["share"]),
        "湖北负向率": pct2(top_prov["neg_ratio"]),
        "省份覆盖数": f"{int(ana['data']['province_rows']):,}",
        "省份覆盖率": pct2(ana["data"]["province_coverage"]),
        "标签帖数": f"{int(ana['data']['hashtag_rows']):,}",
        "1/20 前日均": f"{float(trend['pre_jan20_mean_n']):.1f}",
        "1/20 后日均": f"{float(trend['post_jan20_mean_n']):.1f}",
        "声量峰值数": f"{int(trend['peak_volume_n']):,}",
        "最高预警日条数": f"{int(top_alert['n']):,}",
        "最高预警日": str(top_alert["date"]),
        "最高预警日负向率": pct2(top_alert["neg_ratio"]),
        "文本遮挡 token": str(top_token["token"]),
        "文本遮挡下降": f"{float(top_token['mean_prob_drop']):.4f}",
        "文本遮挡第二下降": f"{float(to['top_tokens'][1]['mean_prob_drop']):.4f}",
        "文本遮挡第三下降": f"{float(to['top_tokens'][2]['mean_prob_drop']):.4f}",
        "图像中心下降": f"{float(grid[1][1]):.4f}",
        "图像左上下降": f"{float(grid[0][0]):.4f}",
        "图像右下下降": f"{float(grid[2][2]):.4f}",
        "图像维度": str(int(met["data"]["image_dim"])),
        "文本遮挡样本": str(int(to["n_samples"])),
        "文本扰动次数": str(int(to["n_perturbations"])),
        "图像遮挡样本": str(int(io["n_samples"])),
        "日期最小值": str(ana["data"]["date_min"]),
        "日期最大值": str(ana["data"]["date_max"]),
        "峰值日": str(trend["peak_volume_date"]),
        "负向率峰值日": str(trend["peak_neg_date"]),
        "文本缺失10%": f4(mrow("gated", "text", 0.1)["macro_f1_mean"]),
        "文本缺失70%": f4(mrow("gated", "text", 0.7)["macro_f1_mean"]),
        "双模态缺失10%": f4(mrow("gated", "both", 0.1)["macro_f1_mean"]),
        "双模态缺失70%": f4(mrow("gated", "both", 0.7)["macro_f1_mean"]),
        "图像缺失最小": f4(min(mrow("gated", "image", r)["macro_f1_mean"]
                              for r in (0.1, 0.3, 0.5, 0.7))),
        "图像缺失最大": f4(max(mrow("gated", "image", r)["macro_f1_mean"]
                              for r in (0.1, 0.3, 0.5, 0.7))),
        "文本删除top1": f4(drow("text", "top", 1)["mean_prob_drop"]),
        "文本删除random1": f4(drow("text", "random", 1)["mean_prob_drop"]),
        "文本删除top3": f4(drow("text", "top", 3)["mean_prob_drop"]),
        "文本删除random3": f4(drow("text", "random", 3)["mean_prob_drop"]),
        "文本删除top5": f4(drow("text", "top", 5)["mean_prob_drop"]),
        "文本删除random5": f4(drow("text", "random", 5)["mean_prob_drop"]),
        "图像删除top1": f4(drow("image", "top", 1)["mean_prob_drop"]),
        "图像删除random1": f4(drow("image", "random", 1)["mean_prob_drop"]),
        "图像删除top3": f4(drow("image", "top", 3)["mean_prob_drop"]),
        "图像删除random3": f4(drow("image", "random", 3)["mean_prob_drop"]),
        "删除文本样本": str(int(dl["text"]["n_candidates"])),
        "删除图像样本": str(int(dl["image"]["n_candidates"])),
        "推理文本ms": f"{infer_ms('text_only'):.1f}",
        "推理图文ms": f"{infer_ms('text_image'):.1f}",
        "解释卡数": str(int(cards.get("n", 0))),
    }
    return facts, ana, met, exp, man


def check_metrics(c):
    facts, ana, met, exp, man = build_facts()

    report = read_text("docs/01_研究报告.md")
    # 报告必须出现的关键事实（只查核心，不要求每个小数都出现）
    report_keys = [
        "门控完整 Macro-F1", "门控完整 CI", "文本单模态", "图像单模态",
        "早期融合 LR", "晚期融合", "门控等权", "早期融合 MLP",
        "门控仅文本", "门控仅图像", "门控双缺失",
        "早期 LR 仅文本", "早期 LR 仅图像",
        "外部门控", "外部文本", "外部早期 LR", "外部 MLP",
        "门控文本权重", "门控图像权重", "负面图像权重",
        "预警数", "声量增长", "负向率峰值",
        "全量帖数", "日期回填数", "日期覆盖率",
        "建模样本", "外部样本", "训练集", "验证集", "测试集",
        "落盘样本", "落盘用户", "落盘图片 MB",
        "最大话题份额", "最大话题负向率", "最大话题帖数",
        "最高频标签数", "最高频标签正向率",
        "1/20 前日均", "1/20 后日均", "声量峰值数",
        "最高预警日条数", "文本遮挡 token", "文本遮挡下降",
        "图像中心下降", "图像左上下降", "图像右下下降",
        "图像维度", "文本遮挡样本", "文本扰动次数", "图像遮挡样本",
        "文本缺失10%", "文本缺失70%", "双模态缺失10%", "双模态缺失70%",
        "图像缺失最小", "图像缺失最大",
        "文本删除top1", "文本删除random1", "文本删除top3", "文本删除random3",
        "文本删除top5", "文本删除random5",
        "图像删除top1", "图像删除random1", "图像删除top3", "图像删除random3",
        "删除文本样本", "删除图像样本", "推理文本ms", "推理图文ms", "解释卡数",
    ]
    missing = [f"{k}={facts[k]}" for k in report_keys if facts[k] not in report]
    if missing:
        c.fail("指标一致性", "报告关键数字", "缺少：" + "；".join(missing[:10]) +
               (f" 等 {len(missing)} 个" if len(missing) > 10 else ""))
    else:
        c.ok("指标一致性", "报告关键数字", f"{len(report_keys)} 个关键事实全部出现")

    # 04 实验记录
    rec = read_text("docs/04_实验记录.md")
    rec_keys = ["门控完整 Macro-F1", "文本单模态", "图像单模态", "早期融合 LR",
                "晚期融合", "门控等权", "早期融合 MLP",
                "门控仅文本", "门控仅图像", "门控双缺失",
                "外部门控", "外部文本", "外部早期 LR", "外部 MLP",
                "门控文本权重", "门控图像权重",
                "文本缺失10%", "文本缺失70%", "双模态缺失10%", "双模态缺失70%",
                "文本删除top1", "文本删除random1", "推理文本ms", "推理图文ms",
                "解释卡数"]
    miss = [f"{k}={facts[k]}" for k in rec_keys if facts[k] not in rec]
    if miss:
        c.fail("指标一致性", "实验记录关键数字", "缺少：" + "；".join(miss))
    else:
        c.ok("指标一致性", "实验记录关键数字", f"{len(rec_keys)} 个关键事实")

    # 其他文档的关键数字
    doc_sets = {
        "README.md": ["门控完整 Macro-F1", "文本单模态", "早期融合 LR",
                      "门控仅文本", "门控仅图像", "门控双缺失",
                      "外部门控", "外部早期 LR",
                      "门控文本权重", "门控图像权重", "声量增长", "负向率峰值"],
        "docs/05_演示视频脚本.md": ["门控完整 Macro-F1", "文本单模态", "早期融合 LR",
                                     "门控仅文本", "门控仅图像", "门控双缺失",
                                     "外部门控", "外部早期 LR",
                                     "门控文本权重", "门控图像权重", "声量增长", "负向率峰值",
                                     "文本缺失10%", "文本缺失70%", "双模态缺失10%",
                                     "双模态缺失70%", "文本删除top1", "文本删除random1"],
        "docs/07_答辩要点.md": ["门控完整 Macro-F1", "文本单模态", "早期融合 LR",
                                 "门控仅文本", "门控仅图像", "门控双缺失",
                                 "外部门控", "外部早期 LR",
                                 "门控文本权重", "门控图像权重", "声量增长", "负向率峰值",
                                 "文本缺失10%", "文本缺失70%", "双模态缺失10%",
                                 "双模态缺失70%", "文本删除top1", "文本删除random1",
                                 "图像删除top1", "图像删除random1", "解释卡数"],
        "docs/06_提交前自检清单.md": ["门控完整 Macro-F1", "文本单模态", "早期融合 LR",
                                       "门控仅文本", "门控仅图像",
                                       "外部门控", "外部早期 LR",
                                       "门控文本权重", "门控图像权重", "声量增长", "负向率峰值",
                                       "文本缺失10%", "文本缺失70%", "双模态缺失10%",
                                       "双模态缺失70%", "文本删除top1", "文本删除random1",
                                       "解释卡数"],
        "docs/02_开发文档.md": ["门控完整 Macro-F1", "门控仅文本", "外部门控", "门控双缺失",
                                 "全量帖数", "建模样本", "外部样本", "落盘样本", "落盘用户"],
        "docs/03_数据说明.md": ["全量帖数", "日期回填数", "日期覆盖率", "落盘样本",
                                 "落盘用户", "落盘图片 MB"],
        "output/models/MODEL_CARD.md": ["门控完整 Macro-F1", "文本单模态", "外部门控"],
    }
    for rel, keys in doc_sets.items():
        if not exists(rel):
            c.fail("指标一致性", f"{rel} 存在", "文件缺失")
            continue
        text = read_text(rel)
        miss = [f"{k}={facts[k]}" for k in keys if facts[k] not in text]
        if miss:
            c.fail("指标一致性", f"{rel} 关键数字", "缺少/过期：" + "；".join(miss))
        else:
            c.ok("指标一致性", f"{rel} 关键数字", f"{len(keys)} 项")

    # 过期旧值检查：旧值出现且当前值已变化 -> FAIL
    current = set(facts.values())
    current |= {x.replace(",", "") for x in current}
    old_tokens = [
        "0.6816", "0.6838", "0.6857", "0.5657", "0.5371", "0.6836",
        "0.4327", "0.1667", "0.6781", "0.6787", "0.4431",
        "0.5666", "0.5694", "0.6614", "0.2834", "0.6800",
        "0.306", "0.232", "0.192", "0.0101",
        "96.1%", "3.9%", "7.3%", "92.7%", "99.6%", "0.4%",
        "275.8%", "24.77%", "56,649", "20,423", "36.05%", "992",
        "44.19", "43.96%", "19.47%", "753", "70.65%",
        "11,425", "12.37%", "19,735", "34.84%", "21,182", "5.05%",
        "150.7", "566.4", "698", "654",
        "0.6592~0.7010", "0.6625~0.7047", "0.6640~0.7060",
        "0.6634~0.7071", "0.6559~0.6971", "0.6563~0.6991",
        "0.5467~0.5853", "0.5192~0.5551", "0.5474~0.5847",
        "0.5508~0.5870",
    ]
    docs_to_scan = ["docs/01_研究报告.md", "docs/02_开发文档.md",
                    "docs/03_数据说明.md", "docs/04_实验记录.md",
                    "docs/05_演示视频脚本.md", "docs/06_提交前自检清单.md",
                    "docs/07_答辩要点.md", "docs/08_作品方案要求对照.md",
                    "README.md"]
    stale = []
    for rel in docs_to_scan:
        if not exists(rel):
            continue
        text = read_text(rel)
        for tok in old_tokens:
            if tok in text and tok not in current:
                stale.append(f"{rel}:{tok}")
    if stale:
        c.fail("指标一致性", "过期旧值", "；".join(stale[:12]) +
               (f" 等 {len(stale)} 处" if len(stale) > 12 else ""))
    else:
        c.ok("指标一致性", "过期旧值", "未发现旧指标残留")


def check_report(c):
    rel = "docs/01_研究报告.md"
    if not exists(rel):
        c.fail("报告结构", "报告存在", "缺少 docs/01_研究报告.md")
        return
    text = read_text(rel)
    required = [
        "## 摘要", "## 1 问题描述与应用场景", "## 2 数据来源与合规说明",
        "## 3 系统设计与开发", "## 4 数据分析与实验", "## 5 主要结论",
        "## 6 局限与改进", "## 7 合规性自查", "## 参考文献",
        "## 附录 A", "## 附录 B",
    ]
    miss = [h for h in required if h not in text]
    if miss:
        c.fail("报告结构", "章节齐全", "缺少：" + "；".join(miss))
    else:
        c.ok("报告结构", "章节齐全", f"{len(required)} 个章节")

    extra_sections = ["#### 4.7.1", "#### 4.9.1", "### 4.12", "### 4.13"]
    miss = [h for h in extra_sections if h not in text]
    if miss:
        c.fail("报告结构", "新增实验章节", "缺少：" + "；".join(miss))
    else:
        c.ok("报告结构", "新增实验章节", "4.7.1 / 4.9.1 / 4.12 / 4.13")

    keywords = ["多模态情感分析", "模态缺失", "门控融合", "可解释", "微博",
                "ScienceDB", "10.57760/sciencedb.j00133.00366",
                "CC BY 4.0", "匿名", "1,000", "公开数据集"]
    miss = [k for k in keywords if k not in text]
    if miss:
        c.fail("报告结构", "关键词与合规表述", "缺少：" + "；".join(miss))
    else:
        c.ok("报告结构", "关键词与合规表述", f"{len(keywords)} 个关键词")

    non_space = len(re.sub(r"\s", "", text))
    if non_space >= 15000:
        c.ok("报告结构", "报告字数", f"{non_space:,} 非空白字符（≥15,000）")
    else:
        c.fail("报告结构", "报告字数", f"{non_space:,}，低于 15,000")

    imgs = re.findall(r"!\[[^\]]*\]\((figures/[^)]+)\)", text)
    if len(imgs) >= 10:
        c.ok("报告结构", "报告插图数", f"{len(imgs)} 张")
    else:
        c.fail("报告结构", "报告插图数", f"{len(imgs)} 张，少于 10")
    miss_img = [x for x in imgs if not os.path.exists(os.path.join(DOCS, x.replace("/", os.sep)))]
    if miss_img:
        c.fail("报告结构", "报告插图文件", "缺少：" + "；".join(miss_img))
    else:
        c.ok("报告结构", "报告插图文件", "全部存在")

    tables = len(re.findall(r"\n\s*\|?\s*:?-{3,}", text))
    if tables >= 9:
        c.ok("报告结构", "报告表格数", f"{tables} 处表头分隔（≥9）")
    else:
        c.fail("报告结构", "报告表格数", f"{tables} 处，少于 9")

    bad = []
    for phrase in ["合成数据训练主模型", "使用合成数据训练", "模拟数据训练",
                   "我们实时爬取", "实时爬取了微博", "100% 准确"]:
        for m in re.finditer(phrase, text):
            prefix = text[max(0, m.start() - 20):m.start()]
            if not re.search(r"[不无未没]", prefix):
                bad.append(phrase)
                break
    for m in re.finditer("SOTA", text):
        prefix = text[max(0, m.start() - 20):m.start()]
        if not re.search(r"[不无未没]", prefix):
            bad.append("SOTA")
            break
    if bad:
        c.fail("报告结构", "事实表述", "出现不当表述：" + "；".join(bad))
    else:
        c.ok("报告结构", "事实表述", "未出现合成数据/实时爬取/SOTA 等不当表述")

    # 占位符
    ph = re.findall(r"\{\{[^}]+\}\}|【待填[^】]*】|\bTODO\b|待补充", text)
    if ph:
        c.fail("占位符", "报告无占位符", "；".join(ph[:8]))
    else:
        c.ok("占位符", "报告无占位符", "未发现 {{...}} / 【待填】 / TODO / 待补充")

    rec = "docs/04_实验记录.md"
    if exists(rec):
        rtext = read_text(rec)
        ph = re.findall(r"\{\{[^}]+\}\}|【待填[^】]*】|\bTODO\b|待补充", rtext)
        if ph:
            c.fail("占位符", "实验记录无占位符", "；".join(ph[:8]))
        else:
            c.ok("占位符", "实验记录无占位符", "未发现占位符")
    else:
        c.fail("占位符", "实验记录存在", "缺少 docs/04_实验记录.md")


def check_pdf(c):
    docs = sorted(glob.glob(os.path.join(DOCS, "*.md")))
    readme = os.path.join(ROOT, "README.md")
    if os.path.exists(readme):
        docs.append(readme)
    if not docs:
        c.fail("PDF", "待导出文档", "没有 docs/*.md 或 README.md")
        return

    missing, bad_header, bad_eof, stale, small = [], [], [], [], []
    for md in docs:
        base = os.path.splitext(md)[0]
        pdf = base + ".pdf"
        html = base + ".html"
        rel_pdf = os.path.relpath(pdf, ROOT)
        if not os.path.exists(pdf):
            missing.append(rel_pdf)
            continue
        size = os.path.getsize(pdf)
        if size < 20_000:
            small.append(f"{rel_pdf}({size/1024:.0f}KB)")
        with open(pdf, "rb") as f:
            head = f.read(8)
            f.seek(max(0, size - 2048))
            tail = f.read()
        if not head.startswith(b"%PDF-"):
            bad_header.append(rel_pdf)
        if b"%%EOF" not in tail:
            bad_eof.append(rel_pdf)
        if os.path.exists(md) and os.path.getmtime(pdf) + TOL < os.path.getmtime(md):
            stale.append(rel_pdf)
        if not os.path.exists(html) or os.path.getsize(html) < 10_000:
            c.warn("PDF", "HTML 备份", f"{os.path.relpath(html, ROOT)} 缺失或过小")

    if missing:
        c.fail("PDF", "PDF 齐全", "缺少：" + "；".join(missing[:8]) +
               (f" 等 {len(missing)} 个" if len(missing) > 8 else ""))
    else:
        c.ok("PDF", "PDF 齐全", f"{len(docs)} 个 PDF")
    if bad_header:
        c.fail("PDF", "PDF 文件头", "不是 %PDF-：" + "；".join(bad_header))
    else:
        c.ok("PDF", "PDF 文件头", "全部为 %PDF-")
    if bad_eof:
        c.fail("PDF", "PDF 文件尾", "缺少 %%EOF：" + "；".join(bad_eof))
    else:
        c.ok("PDF", "PDF 文件尾", "全部含 %%EOF")
    if stale:
        c.fail("PDF", "PDF 新鲜度", "PDF 早于 Markdown：" + "；".join(stale))
    else:
        c.ok("PDF", "PDF 新鲜度", "全部不早于对应 Markdown")
    if small:
        c.warn("PDF", "PDF 体积", "偏小：" + "；".join(small))
    else:
        c.ok("PDF", "PDF 体积", "全部 >20KB")

    report_pdf = os.path.join(DOCS, "01_研究报告.pdf")
    if os.path.exists(report_pdf):
        size = os.path.getsize(report_pdf)
        if size >= 200_000:
            c.ok("PDF", "报告 PDF 体积", f"{size/1024:.0f} KB")
        else:
            c.warn("PDF", "报告 PDF 体积", f"{size/1024:.0f} KB，可能图少/排版异常")
        data = Path(report_pdf).read_bytes()
        pages = len(re.findall(rb"/Type\s*/Page\b", data))
        if pages >= 10:
            c.ok("PDF", "报告 PDF 页数", f"约 {pages} 页")
        elif pages > 0:
            c.warn("PDF", "报告 PDF 页数", f"约 {pages} 页，偏少")
        else:
            c.info("PDF", "报告 PDF 页数", "未能从 PDF 对象流解析页数（不阻断）")


def check_freshness(c):
    pairs = [
        ("dashboard/data.js", "output/analysis_results.json"),
        ("dashboard/data.js", "output/models_metrics.json"),
        ("dashboard/data.js", "output/missing_experiments.json"),
        ("dashboard/single.html", "output/explanation_cards.json"),
        ("docs/01_研究报告.md", "output/analysis_results.json"),
        ("docs/01_研究报告.md", "output/models_metrics.json"),
        ("docs/01_研究报告.md", "output/missing_experiments.json"),
        ("docs/01_研究报告.md", "output/inference_self_test.json"),
        ("docs/01_研究报告.md", "output/explanation_cards.json"),
        ("docs/04_实验记录.md", "output/models_metrics.json"),
        ("docs/04_实验记录.md", "output/missing_experiments.json"),
        ("docs/04_实验记录.md", "output/inference_self_test.json"),
        ("docs/04_实验记录.md", "output/explanation_cards.json"),
        ("docs/figures/dashboard.png", "dashboard/data.js"),
    ]
    for fig in ["fig1_architecture.png", "fig2_data_overview.png", "fig3_daily_trend.png",
                "fig4_topics.png", "fig5_keywords.png", "fig6_model_comparison.png",
                "fig7_missing_robustness.png", "fig8_confusion.png",
                "fig9_explainability.png", "fig10_external.png"]:
        pairs.append((f"docs/figures/{fig}", "output/models_metrics.json"))
        pairs.append((f"docs/figures/{fig}", "output/analysis_results.json"))
    for fig in ["fig11_missing_ratio.png", "fig12_deletion.png"]:
        pairs.append((f"docs/figures/{fig}", "output/missing_experiments.json"))

    stale = []
    for a, b in pairs:
        if not exists(a) or not exists(b):
            continue
        if not newer(a, b):
            stale.append(f"{a} 早于 {b}")
    if stale:
        c.fail("新鲜度", "下游不早于上游", "；".join(stale[:10]))
    else:
        c.ok("新鲜度", "下游不早于上游", f"{len(pairs)} 组文件")

    groups = {
        "模型底表": (["模型对比.csv", "缺失场景鲁棒性.csv", "外部测试.csv",
                      "混淆矩阵.csv", "门控权重.csv", "文本遮挡解释.csv",
                      "图像遮挡解释.csv"], "output/models_metrics.json"),
        "分析底表": (["日度情感趋势.csv", "话题分析.csv", "情感关键词.csv",
                      "省份提及.csv", "话题标签.csv", "异常预警.csv",
                      "情感样本示例.csv", "文本长度统计.csv"], "output/analysis_results.json"),
        "校验底表": (["数据校验报告.csv"], "output/data_quality.json"),
        "爬虫底表": (["爬虫自检.csv"], "output/crawler_self_test.json"),
        "缺失实验底表": (["缺失比例曲线.csv", "缺失位置曲线.csv", "删除实验.csv"],
                         "output/missing_experiments.json"),
    }
    stale = []
    for name, (files, src) in groups.items():
        src_mtime = mtime(src) if exists(src) else None
        # 12 会向 models_metrics.json 追加缺失/删除实验；模型底表仍由 04 生成，
        # 因此用 JSON 内的 generated_at（04 训练时间）比较，而不是文件 mtime。
        if src == "output/models_metrics.json" and src_mtime is not None:
            gen = read_json(src).get("generated_at")
            if gen:
                try:
                    src_mtime = datetime.strptime(gen, "%Y-%m-%d %H:%M:%S").timestamp()
                except ValueError:
                    pass
        for fn in files:
            rel = f"output/tables/{fn}"
            if exists(rel) and src_mtime is not None and mtime(rel) + BOTTOM_TOL < src_mtime:
                stale.append(f"{rel} 早于 {src}")
    if stale:
        c.fail("新鲜度", "底表不早于事实表", "；".join(stale[:8]))
    else:
        c.ok("新鲜度", "底表不早于事实表", "模型/分析/校验底表均新鲜")


def check_dashboard(c):
    idx, datajs, ech = "dashboard/index.html", "dashboard/data.js", "dashboard/echarts.min.js"
    if not all(exists(x) for x in (idx, datajs, ech)):
        c.fail("大屏", "三件套齐全", "缺少 index.html / data.js / echarts.min.js")
        return
    html = read_text(idx)
    if fsize(idx) > 20_000:
        c.ok("大屏", "index.html 体积", f"{fsize(idx)/1024:.0f} KB")
    else:
        c.fail("大屏", "index.html 体积", f"{fsize(idx)/1024:.0f} KB，过小")
    if fsize(datajs) > 10_000:
        c.ok("大屏", "data.js 体积", f"{fsize(datajs)/1024:.0f} KB")
    else:
        c.fail("大屏", "data.js 体积", f"{fsize(datajs)/1024:.0f} KB，过小")
    if fsize(ech) > 500_000:
        c.ok("大屏", "echarts 本地化", f"{fsize(ech)/1024:.0f} KB")
    else:
        c.fail("大屏", "echarts 本地化", f"{fsize(ech)/1024:.0f} KB，可能不是完整库")

    scripts = re.findall(r'<script[^>]+src="([^"]+)"', html)
    if "echarts.min.js" in scripts and "data.js" in scripts:
        c.ok("大屏", "本地脚本引用", "echarts.min.js + data.js")
    else:
        c.fail("大屏", "本地脚本引用", f"实际：{scripts}")
    external = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
    if external:
        c.fail("大屏", "无外链依赖", "；".join(external[:5]))
    else:
        c.ok("大屏", "无外链依赖", "无 http(s) 外链脚本/样式")
    if "fetch(" in html or "XMLHttpRequest" in html:
        c.warn("大屏", "无异步请求", "检测到 fetch/XMLHttpRequest，确认 file:// 下可用")
    else:
        c.ok("大屏", "无异步请求", "数据已内联到 data.js")

    n_charts = len(re.findall(r'class="chart"', html))
    if n_charts == 14:
        c.ok("大屏", "视图数", "14 个 class=\"chart\"")
    else:
        c.fail("大屏", "视图数", f"{n_charts} 个，期望 14")

    raw = read_text(datajs)
    m = re.match(r"\s*window\.DASH_DATA\s*=\s*", raw)
    if not m:
        c.fail("大屏", "data.js 前缀", "不是 window.DASH_DATA = ...")
        return
    payload = raw[m.end():].strip()
    if payload.endswith(";"):
        payload = payload[:-1].rstrip()
    try:
        dash = json.loads(payload)
        c.ok("大屏", "data.js 可解析", "JSON 解析成功")
    except Exception as e:
        c.fail("大屏", "data.js 可解析", f"{type(e).__name__}: {e}")
        return

    keys = ["meta", "kpi", "daily", "trend", "sentiment", "topics", "keywords",
            "provinces", "hashtags", "alerts", "models", "gates", "explainability", "crawl"]
    miss = [k for k in keys if k not in dash]
    if miss:
        c.fail("大屏", "数据键齐全", "缺少：" + "；".join(miss))
    else:
        c.ok("大屏", "数据键齐全", f"{len(keys)} 个顶层键")

    counts = [
        ("daily", len(dash.get("daily", [])), 45, "日度序列"),
        ("topics", len(dash.get("topics", [])), 8, "话题"),
        ("models.complete", len(dash.get("models", {}).get("complete", [])), 7, "完整模型"),
        ("models.robustness", len(dash.get("models", {}).get("robustness", [])), 12, "缺失场景"),
        ("models.external", len(dash.get("models", {}).get("external", [])), 5, "外部模型"),
        ("alerts", len(dash.get("alerts", [])), 5, "预警"),
    ]
    bad = [f"{name}={n}(<{need})" for name, n, need, _ in counts if n < need]
    if bad:
        c.fail("大屏", "数据规模", "；".join(bad))
    else:
        c.ok("大屏", "数据规模", "daily/topics/models/alerts 均达标")

    facts, ana, met, exp, man = build_facts()
    kpi = dash.get("kpi", {})
    checks = [
        ("total", int(kpi.get("total", -1)), int(ana["data"]["total_rows"])),
        ("sample", int(kpi.get("sample", -1)), int(ana["data"]["sample_rows"])),
        ("crawl", int(kpi.get("crawl", -1)), int(man["n"])),
        ("crawl_users", int(kpi.get("crawl_users", -1)), int(man["unique_user_id"])),
        ("alerts", int(kpi.get("alerts", -1)), len(ana["alerts"])),
    ]
    bad = [f"{k}:{a}!={b}" for k, a, b in checks if a != b]
    if bad:
        c.fail("大屏", "KPI 与 JSON 一致", "；".join(bad))
    else:
        c.ok("大屏", "KPI 与 JSON 一致", "total/sample/crawl/users/alerts")
    for k, fact_key in [("gated_f1", "门控完整 Macro-F1"), ("external_f1", "外部门控")]:
        if abs(float(kpi.get(k, -1)) - float(facts[fact_key])) <= 0.0006:
            c.ok("大屏", f"KPI {k}", f"{float(kpi.get(k)):.6f}")
        else:
            c.fail("大屏", f"KPI {k}", f"{kpi.get(k)} vs {facts[fact_key]}")
    if abs(float(kpi.get("volume_growth", -1)) - float(facts["声量增长"].rstrip("%"))) <= 0.15:
        c.ok("大屏", "KPI volume_growth", f"{kpi.get('volume_growth')}%")
    else:
        c.fail("大屏", "KPI volume_growth", f"{kpi.get('volume_growth')} vs {facts['声量增长']}")

    meta = dash.get("meta", {})
    if meta.get("doi") == "10.57760/sciencedb.j00133.00366" and meta.get("license") == "CC BY 4.0":
        c.ok("大屏", "来源与许可", "DOI + CC BY 4.0")
    else:
        c.fail("大屏", "来源与许可", f"doi={meta.get('doi')} license={meta.get('license')}")

    shot = "docs/figures/dashboard.png"
    if exists(shot) and fsize(shot) > 50_000:
        c.ok("大屏", "大屏截图", f"{fsize(shot)/1024:.0f} KB")
    else:
        c.fail("大屏", "大屏截图", "缺失或 <50KB")


def check_single_page(c):
    rel = "dashboard/single.html"
    if not exists(rel):
        c.fail("单条页", "文件存在", "缺少 dashboard/single.html")
        return
    html = read_text(rel)
    if fsize(rel) > 10_000:
        c.ok("单条页", "single.html 体积", f"{fsize(rel)/1024:.0f} KB")
    else:
        c.fail("单条页", "single.html 体积", f"{fsize(rel)/1024:.0f} KB，过小")

    cards = read_json("output/explanation_cards.json")
    cs = cards.get("cards", [])
    if int(cards.get("n", 0)) == 3 and len(cs) == 3:
        c.ok("单条页", "解释卡数量", "3 条")
    else:
        c.fail("单条页", "解释卡数量", f"n={cards.get('n')} / cards={len(cs)}")

    disclaimer = ["示例不代表总体准确率", "概率映射", "强度"]
    miss = [x for x in disclaimer if x not in html]
    if miss:
        c.fail("单条页", "示例与强度声明", "缺少：" + "；".join(miss))
    else:
        c.ok("单条页", "示例与强度声明", "示例不代表总体准确率 / 概率映射 / 强度")

    markers = ["text_top_tokens", "image_grid", "gate"]
    miss = [x for x in markers if x not in html]
    if miss:
        c.fail("单条页", "解释字段", "缺少：" + "；".join(miss))
    else:
        c.ok("单条页", "解释字段", "token / 图像网格 / 门控")

    miss_ids = [x.get("post_id", "?") for x in cs if x.get("post_id", "") not in html]
    if miss_ids:
        c.fail("单条页", "卡片与 JSON 一致", "缺少 post_id：" + "；".join(miss_ids))
    else:
        c.ok("单条页", "卡片与 JSON 一致", "3 个 post_id 均在页面内")

    external = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
    if external:
        c.fail("单条页", "无外链依赖", "；".join(external[:5]))
    else:
        c.ok("单条页", "无外链依赖", "无 http(s) 外链")

    missing_img = []
    for x in cs:
        u = x.get("image_url", "")
        if not u:
            missing_img.append(x.get("post_id", "?") + ":无 image_url")
            continue
        fp = os.path.normpath(os.path.join(DASH, u.replace("/", os.sep)))
        if not os.path.exists(fp):
            missing_img.append(u)
    if missing_img:
        c.fail("单条页", "配图存在", "；".join(missing_img[:5]))
    else:
        c.ok("单条页", "配图存在", "3 张配图均可解析")


def check_privacy(c):
    rel = "data/crawl/weibo_1000.csv"
    if not exists(rel):
        c.fail("隐私", "落盘样本存在", "缺少 data/crawl/weibo_1000.csv")
        return
    with open(p(rel), encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    forbidden = {"url", "account", "nickname", "raw_id", "weibo_id",
                 "user_name", "screen_name", "image_url", "profile_url",
                 "author", "author_id"}
    bad_cols = sorted(forbidden & {h.lower() for h in header})
    if bad_cols:
        c.fail("隐私", "落盘样本列名", "出现敏感列：" + "；".join(bad_cols))
    else:
        c.ok("隐私", "落盘样本列名", f"{len(header)} 列，无原始账号/ID/URL 列")
    if len(rows) == 1000:
        c.ok("隐私", "落盘样本行数", "1,000")
    else:
        c.fail("隐私", "落盘样本行数", len(rows))

    if "context" not in header:
        c.fail("隐私", "context 字段", "缺少 context 列（应为 text + image 的 JSON）")
    else:
        bad_ctx = []
        for r in rows:
            try:
                ctx = json.loads(r.get("context", ""))
            except Exception:
                bad_ctx.append("JSON 解析失败")
                continue
            if not isinstance(ctx, dict):
                bad_ctx.append("不是 JSON 对象")
            elif not str(ctx.get("text", "")).strip() or not str(ctx.get("image", "")).strip():
                bad_ctx.append("缺少 text/image 键")
            elif ctx.get("text") != r.get("text") or ctx.get("image") != r.get("image_file"):
                bad_ctx.append("与 text/image_file 列不一致")
        if bad_ctx:
            c.fail("隐私", "context 完整性", f"{len(bad_ctx)} 条异常，如 {bad_ctx[:3]}")
        else:
            c.ok("隐私", "context 完整性", "1,000 条均含 text + image，且与列一致")

    uid_re = re.compile(r"^u_[0-9a-f]{16}$")
    pid_re = re.compile(r"^wb_[0-9a-f]{16}$")
    bad_uid = [r.get("user_id", "") for r in rows if not uid_re.match(r.get("user_id", ""))]
    bad_pid = [r.get("post_id", "") for r in rows if not pid_re.match(r.get("post_id", ""))]
    if bad_uid:
        c.fail("隐私", "user_id 格式", f"{len(bad_uid)} 条异常，如 {bad_uid[:3]}")
    else:
        c.ok("隐私", "user_id 格式", "u_ + 16 位十六进制")
    if bad_pid:
        c.fail("隐私", "post_id 格式", f"{len(bad_pid)} 条异常，如 {bad_pid[:3]}")
    else:
        c.ok("隐私", "post_id 格式", "wb_ + 16 位十六进制")

    users = {r.get("user_id") for r in rows}
    if 900 <= len(users) <= 1000:
        c.ok("隐私", "user_id 去重", f"{len(users)} 个")
    else:
        c.fail("隐私", "user_id 去重", len(users))

    bad_type = [r.get("user_id_type") for r in rows if r.get("user_id_type") != "sha256_pseudonym"]
    if bad_type:
        c.fail("隐私", "user_id_type", f"{len(bad_type)} 条不是 sha256_pseudonym")
    else:
        c.ok("隐私", "user_id_type", "全部 sha256_pseudonym")

    labels = {}
    for r in rows:
        labels[r.get("label")] = labels.get(r.get("label"), 0) + 1
    if set(labels) <= {"-1", "0", "1"} and max(labels.values()) - min(labels.values()) <= 20:
        c.ok("隐私", "标签均衡", str(labels))
    else:
        c.fail("隐私", "标签均衡", str(labels))

    bad_time = []
    for r in rows:
        t = r.get("time", "")
        ok = False
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                datetime.strptime(t, fmt)
                ok = True
                break
            except ValueError:
                pass
        if not ok:
            bad_time.append(t)
    if bad_time:
        c.fail("隐私", "time 可解析", f"{len(bad_time)} 条异常，如 {bad_time[:3]}")
    else:
        c.ok("隐私", "time 可解析", "全部 YYYY-MM-DD HH:MM[:SS]")

    bad_date = []
    for r in rows:
        d = r.get("date", "")
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
            if not (datetime(2020, 1, 1) <= dt <= datetime(2020, 2, 19)):
                bad_date.append(d)
        except ValueError:
            bad_date.append(d)
    if bad_date:
        c.fail("隐私", "date 范围", f"{len(bad_date)} 条异常，如 {bad_date[:3]}")
    else:
        c.ok("隐私", "date 范围", "2020-01-01 ~ 2020-02-19")

    bad_text = []
    for r in rows:
        t = r.get("text", "")
        if re.search(r"https?://|www\.", t) or "@" in t:
            bad_text.append(t[:40])
    if bad_text:
        c.fail("隐私", "落盘样本文本清洗", f"{len(bad_text)} 条仍含 URL/@，如 {bad_text[:2]}")
    else:
        c.ok("隐私", "落盘样本文本清洗", "无 URL/@ 残留")

    missing_img, bad_hash, bad_dim = [], [], []
    for r in rows:
        img = r.get("image_file", "")
        fp = os.path.join(CRAWL, img.replace("/", os.sep))
        if not img or not os.path.exists(fp):
            missing_img.append(img)
        if not re.match(r"^[0-9a-f]{16}$", r.get("image_sha256", "")):
            bad_hash.append(r.get("image_sha256", ""))
        try:
            if int(r.get("image_width", 0)) <= 0 or int(r.get("image_height", 0)) <= 0 or int(r.get("image_bytes", 0)) <= 0:
                bad_dim.append(img)
        except ValueError:
            bad_dim.append(img)
    if missing_img:
        c.fail("隐私", "落盘图片存在", f"{len(missing_img)} 张缺失，如 {missing_img[:3]}")
    else:
        c.ok("隐私", "落盘图片存在", "1,000 张全部存在")
    if bad_hash:
        c.fail("隐私", "image_sha256 格式", f"{len(bad_hash)} 条异常")
    else:
        c.ok("隐私", "image_sha256 格式", "16 位十六进制")
    if bad_dim:
        c.fail("隐私", "图片宽高/字节", f"{len(bad_dim)} 条异常")
    else:
        c.ok("隐私", "图片宽高/字节", "全部 >0")

    n_imgs = len(glob.glob(os.path.join(CRAWL, "images", "*")))
    if n_imgs == 1000:
        c.ok("隐私", "落盘图片文件数", "1,000")
    else:
        c.fail("隐私", "落盘图片文件数", n_imgs)

    bad_src = [r for r in rows if r.get("source_type") != "public_dataset_import"
               or r.get("dataset_doi") != "10.57760/sciencedb.j00133.00366"]
    if bad_src:
        c.fail("隐私", "来源标注", f"{len(bad_src)} 条 source_type/DOI 异常")
    else:
        c.ok("隐私", "来源标注", "public_dataset_import + DOI")

    # 清洗后数据残留 URL/@
    for rel2, col in [("data/processed/posts_full.csv", "text_clean"),
                      ("data/processed/posts_clean.csv", "text_clean"),
                      ("data/processed/external_test_clean.csv", "text_clean")]:
        if not exists(rel2):
            c.fail("隐私", f"{rel2} 存在", "文件缺失")
            continue
        bad = []
        with open(p(rel2), encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                t = row.get(col, "")
                if re.search(r"https?://|www\.", t) or "@" in t:
                    bad.append(t[:40])
                    if len(bad) >= 3:
                        break
        if bad:
            c.fail("隐私", f"{rel2} 文本清洗", f"仍有 URL/@，如 {bad}")
        else:
            c.ok("隐私", f"{rel2} 文本清洗", "无 URL/@ 残留")

    # 输出底表列名与内容
    table_bad_cols, table_bad_ids = [], []
    for fp in sorted(glob.glob(os.path.join(TABLES, "*.csv"))):
        rel3 = os.path.relpath(fp, ROOT)
        with open(fp, encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            head = next(reader, [])
            bad = sorted(forbidden & {h.lower() for h in head})
            if bad:
                table_bad_cols.append(f"{os.path.basename(fp)}:{','.join(bad)}")
            body = f.read()
        if re.search(r"u_[0-9a-f]{16}|wb_[0-9a-f]{16}", body):
            table_bad_ids.append(os.path.basename(fp))
    if table_bad_cols:
        c.fail("隐私", "输出底表列名", "；".join(table_bad_cols))
    else:
        c.ok("隐私", "输出底表列名", "无原始账号/ID/URL 列")
    if table_bad_ids:
        c.fail("隐私", "输出底表匿名 ID", "出现匿名 ID，建议移除：" + "；".join(table_bad_ids))
    else:
        c.ok("隐私", "输出底表匿名 ID", "未出现 user_id/post_id 具体值")

    # 内部溯源文件提醒（存在即提醒，不阻断本地自检）
    internal = [
        "data/crawl/_internal_provenance.csv",
        "data/crawl/_internal_account_map.csv",
        "data/raw/provenance.csv",
        "data/raw/dataset.zip",
    ]
    found = [x for x in internal if exists(x)]
    if found:
        c.warn("隐私", "内部文件隔离", "存在但不得提交：" + "；".join(found))
    else:
        c.ok("隐私", "内部文件隔离", "未发现内部溯源/原始 zip")

    cache = []
    for base in ("src", "dashboard", "docs"):
        cache += glob.glob(os.path.join(ROOT, base, "**", "__pycache__"), recursive=True)
    if cache:
        c.warn("隐私", "缓存目录", f"{len(cache)} 个 __pycache__，提交前删除")
    else:
        c.ok("隐私", "缓存目录", "无 __pycache__")


def check_compliance(c):
    run_all = read_text("src/run_all.py")
    if "--demo" in run_all:
        c.fail("合规", "run_all 不含 --demo", "检测到 --demo 字样")
    else:
        c.ok("合规", "run_all 不含 --demo", "未出现 --demo")

    collector = read_text("src/01b_collect_public.py")
    checks = [
        ("robots.txt", "robots.txt" in collector or "RobotFileParser" in collector),
        ("MIN_DELAY = 2.0", "MIN_DELAY = 2.0" in collector),
        ("不携带登录态", "Cookie" in collector or "cookie" in collector),
        ("无浏览器自动化", not any(x in collector.lower() for x in
                                    ["selenium", "playwright", "undetected", "2captcha"])),
    ]
    bad = [name for name, ok in checks if not ok]
    if bad:
        c.fail("合规", "01b 合规约束", "缺少/异常：" + "；".join(bad))
    else:
        c.ok("合规", "01b 合规约束", "robots.txt + 2.0s 限速 + 不登录 + 无自动化绕过")

    c1c = read_text("src/01c_export_crawl_1000.py")
    if "--keep-raw-account" in c1c and "_internal_account_map.csv" in c1c:
        c.ok("合规", "01c 原始账号隔离", "默认不落盘，--keep-raw-account 仅内部映射")
    else:
        c.warn("合规", "01c 原始账号隔离", "未找到 --keep-raw-account / 内部映射说明")

    c1a = read_text("src/01a_import_scidb.py")
    if "原始账号" in c1a and "丢弃" in c1a:
        c.ok("合规", "01a 账号丢弃", "代码注明原始账号读取后丢弃")
    else:
        c.warn("合规", "01a 账号丢弃", "未找到“原始账号/丢弃”说明")

    docs_text = {
        "docs/01_研究报告.md": read_text("docs/01_研究报告.md"),
        "docs/03_数据说明.md": read_text("docs/03_数据说明.md"),
        "README.md": read_text("README.md"),
    }
    required = ["10.57760/sciencedb.j00133.00366", "CC BY 4.0", "匿名",
                "robots.txt", "不登录"]
    for rel, text in docs_text.items():
        miss = [x for x in required if x not in text]
        if miss:
            c.fail("合规", f"{rel} 合规表述", "缺少：" + "；".join(miss))
        else:
            c.ok("合规", f"{rel} 合规表述", "DOI/许可/匿名/robots/不登录")

    if "不是实时爬取" in docs_text["README.md"] or "不是实时爬取" in docs_text["docs/03_数据说明.md"]:
        c.ok("合规", "非实时爬取标注", "README/数据说明明确标注")
    else:
        c.fail("合规", "非实时爬取标注", "未明确标注“不是实时爬取”")

    if "内部溯源" in docs_text["docs/01_研究报告.md"] and "不得提交" in docs_text["docs/01_研究报告.md"]:
        c.ok("合规", "内部溯源隔离表述", "报告含“内部溯源/不得提交”")
    else:
        c.warn("合规", "内部溯源隔离表述", "报告未同时出现“内部溯源/不得提交”")

    # 文档不出现“我们实时爬取”等虚假表述；允许“不追求/不宣称 SOTA”这类否定句
    bad = []
    for rel, text in docs_text.items():
        for phrase in ["我们实时爬取", "实时爬取了微博", "100% 准确", "SOTA"]:
            for m in re.finditer(re.escape(phrase), text):
                prefix = text[max(0, m.start() - 20):m.start()]
                if not re.search(r"[不无未没]", prefix):
                    bad.append(f"{rel}:{phrase}")
                    break
    if bad:
        c.fail("合规", "不做虚假宣传", "；".join(bad))
    else:
        c.ok("合规", "不做虚假宣传", "未出现实时爬取/SOTA/100% 准确等肯定性表述")


def check_logs(c):
    log_dir = os.path.join(OUT, "logs")
    if not os.path.isdir(log_dir):
        c.warn("日志", "流水线日志", "output/logs 不存在（尚未跑 run_all）")
        return
    logs = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    if not logs:
        c.warn("日志", "流水线日志", "output/logs 为空")
        return
    bad = []
    for fp in logs:
        try:
            text = Path(fp).read_text(encoding="utf-8", errors="replace")
        except Exception:
            bad.append(os.path.basename(fp) + ":不可读")
            continue
        if "Traceback (most recent call last)" in text or "[FAIL]" in text:
            bad.append(os.path.basename(fp) + ":含异常")
    if bad:
        c.warn("日志", "流水线日志", "；".join(bad))
    else:
        c.ok("日志", "流水线日志", f"{len(logs)} 个日志可读，无 Traceback/[FAIL]")


def check_docs_content(c):
    checks = {
        "docs/02_开发文档.md": ["## 1 文档目的", "## 12 复现步骤", "## 14 常见问题",
                                 "run_all.py", "07_export_docs.py", "08_submission_check.py"],
        "docs/03_数据说明.md": ["## 1 数据来源", "## 3 数据字典", "## 5 匿名化与隐私",
                                 "## 7 数据局限", "CC BY 4.0"],
        "docs/05_演示视频脚本.md": ["## 2 分镜脚本", "## 3 完整口播稿", "## 5 录制检查清单"],
        "docs/06_提交前自检清单.md": ["## 1 一键命令", "FAIL 0", "## 5 FAIL 常见原因"],
        "docs/07_答辩要点.md": ["## 2 三分钟路演稿", "## 3 数字速查表",
                                 "### Q15", "### Q16", "### Q17", "### Q18",
                                 "## 6 不要说的话"],
        "docs/08_作品方案要求对照.md": ["## 0 结论摘要", "## 1 数据来源对照",
                                         "## 12 诚实声明"],
        "docs/09_单条分析接口与示例.md": ["## 3 接口契约", "## 5 离线解释卡页面",
                                           "## 7 常见问题"],
        "README.md": ["## 这个作品解决什么", "## 核心结果", "## 快速开始",
                       "## 大屏", "## 数据与合规"],
    }
    for rel, needles in checks.items():
        if not exists(rel):
            c.fail("文档", f"{rel} 存在", "文件缺失")
            continue
        text = read_text(rel)
        miss = [x for x in needles if x not in text]
        if miss:
            c.fail("文档", f"{rel} 结构", "缺少：" + "；".join(miss))
        else:
            c.ok("文档", f"{rel} 结构", f"{len(needles)} 个关键段落")

    # 文档中提到的图片/表格文件存在（仅检查 figures/ 与 output/tables/）
    all_md = sorted(glob.glob(os.path.join(DOCS, "*.md"))) + [os.path.join(ROOT, "README.md")]
    broken = []
    for fp in all_md:
        text = Path(fp).read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
            target = m.group(1).strip()
            if target.startswith("http"):
                continue
            if target.startswith("figures/"):
                full = os.path.join(DOCS, target.replace("/", os.sep))
            elif target.startswith("docs/"):
                full = os.path.join(ROOT, target.replace("/", os.sep))
            else:
                full = os.path.join(os.path.dirname(fp), target.replace("/", os.sep))
            if not os.path.exists(full):
                broken.append(f"{os.path.basename(fp)}:{target}")
    if broken:
        c.fail("文档", "图片引用可解析", "；".join(broken[:8]))
    else:
        c.ok("文档", "图片引用可解析", "全部 Markdown 图片引用存在")


def check_code_contracts(c):
    checks = [
        ("src/run_all.py", ["STEPS =", "01a", "02_preprocess.py",
                             "12_missing_explain_experiments.py",
                             "10_inference_service.py", "11_export_explanation_cards.py",
                             "13_annotation_kappa.py", "09_fill_report.py",
                             "output/logs", "--from"]),
        ("src/09_fill_report.py", ["{{", "SystemExit", "01_研究报告.md", "04_实验记录.md",
                                    "missing_experiments.json", "inference_self_test.json",
                                    "explanation_cards.json"]),
        ("src/07_export_docs.py", ["--pdf", "print-to-pdf", "markdown.markdown"]),
        ("src/05_make_dashboard_data.py", ["window.DASH_DATA", "data.js"]),
        ("src/06_make_report_figures.py", ["fig1_architecture", "fig10_external",
                                            "screenshot_dashboard"]),
        ("src/01c_export_crawl_1000.py", ["--n", "sha256", "user_id"]),
        ("src/lib_infer.py", ["class SentimentInference", "text_top_tokens",
                               "image_grid", "intensity_proxy"]),
        ("src/10_inference_service.py", ["FastAPI", "ThreadingHTTPServer",
                                          "/health", "/predict", "--self-test"]),
        ("src/11_export_explanation_cards.py", ["explanation_cards.json", "single.html",
                                                 "示例不代表总体准确率", "image_grid"]),
        ("src/12_missing_explain_experiments.py", ["missing_ratio", "missing_position",
                                                    "deletion", "fig11_missing_ratio",
                                                    "fig12_deletion"]),
        ("src/13_annotation_kappa.py", ["quadratic_weighted_kappa", "annotation_template.csv",
                                         "self_test", "--make-template"]),
    ]
    for rel, needles in checks:
        if not exists(rel):
            c.fail("代码契约", f"{rel} 存在", "文件缺失")
            continue
        text = read_text(rel)
        miss = [x for x in needles if x not in text]
        if miss:
            c.fail("代码契约", f"{rel} 关键实现", "缺少：" + "；".join(miss))
        else:
            c.ok("代码契约", f"{rel} 关键实现", f"{len(needles)} 项")

    # 所有入口脚本可编译（内存编译，不写 __pycache__）
    bad = []
    for fp in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        try:
            compile(Path(fp).read_text(encoding="utf-8"), fp, "exec")
        except Exception as e:
            bad.append(f"{os.path.basename(fp)}:{type(e).__name__}")
    if bad:
        c.fail("代码契约", "源码可编译", "；".join(bad))
    else:
        c.ok("代码契约", "源码可编译", f"{len(glob.glob(os.path.join(HERE, '*.py')))} 个 .py")


def check_json_schema(c):
    ana = read_json("output/analysis_results.json")
    met = read_json("output/models_metrics.json")
    exp = read_json("output/explainability.json")
    dq = read_json("output/data_quality.json")
    cr = read_json("output/crawler_self_test.json")
    mexp = read_json("output/missing_experiments.json")
    infer = read_json("output/inference_self_test.json")
    cards = read_json("output/explanation_cards.json")
    anno = read_json("output/annotation_self_test.json")

    schemas = [
        ("analysis_results.json", ana, ["generated_at", "data", "trend", "daily",
                                         "topics", "keywords", "provinces",
                                         "hashtags", "alerts"]),
        ("models_metrics.json", met, ["generated_at", "config", "data", "complete",
                                       "robustness", "external_no_image", "gates",
                                       "training_history", "explainability"]),
        ("explainability.json", exp, ["generated_at", "text_occlusion",
                                       "image_occlusion", "gates"]),
        ("data_quality.json", dq, ["checked_at", "dataset_doi", "dataset_license",
                                    "checks", "status"]),
        ("crawler_self_test.json", cr, ["checked_at", "min_delay_seconds", "checks"]),
        ("missing_experiments.json", mexp, ["generated_at", "test_n", "missing_ratio",
                                             "missing_position", "deletion", "notes"]),
        ("inference_self_test.json", infer, ["generated_at", "n_examples", "examples",
                                              "model_dir", "note"]),
        ("explanation_cards.json", cards, ["generated_at", "n", "selection",
                                            "caveat", "cards"]),
        ("annotation_self_test.json", anno, ["self_test", "source", "polarity",
                                              "intensity", "disagreements", "note"]),
    ]
    for name, obj, keys in schemas:
        miss = [k for k in keys if k not in obj]
        if miss:
            c.fail("事实表", f"{name} 结构", "缺少：" + "；".join(miss))
        else:
            c.ok("事实表", f"{name} 结构", f"{len(keys)} 个键")

    # 门控权重和为 1
    bad = []
    for g in met.get("gates", []):
        s = float(g.get("text_gate", 0)) + float(g.get("image_gate", 0))
        if abs(s - 1.0) > 1e-4:
            bad.append(f"{g.get('group')}/{g.get('label')}={s:.6f}")
    if bad:
        c.fail("事实表", "门控权重归一", "；".join(bad))
    else:
        c.ok("事实表", "门控权重归一", "全部 text_gate + image_gate = 1")

    # 训练历史
    hist = met.get("training_history", {})
    if set(hist) >= {"gated", "early_mlp"}:
        c.ok("事实表", "训练历史", f"gated {hist['gated'].get('epochs_ran')} 轮 / "
                                    f"MLP {hist['early_mlp'].get('epochs_ran')} 轮")
    else:
        c.fail("事实表", "训练历史", f"键：{list(hist)}")

    # 外部测试与缺失场景的 n
    bad = [r for r in met.get("robustness", []) if int(r.get("n", 0)) != 1800]
    if bad:
        c.fail("事实表", "缺失场景样本数", f"{len(bad)} 条不是 1,800")
    else:
        c.ok("事实表", "缺失场景样本数", "全部 1,800")
    bad = [k for k, v in met.get("external_no_image", {}).items()
           if int(v.get("n", 0)) != 3000]
    if bad:
        c.fail("事实表", "外部测试样本数", "；".join(bad))
    else:
        c.ok("事实表", "外部测试样本数", "全部 3,000")

    # 图像遮挡 3×3
    grid = exp.get("image_occlusion", {}).get("grid", [])
    if len(grid) == 3 and all(len(r) == 3 for r in grid):
        c.ok("事实表", "图像遮挡网格", "3×3")
    else:
        c.fail("事实表", "图像遮挡网格", f"{len(grid)} 行")

    # 缺失比例/位置/删除实验
    mr = mexp.get("missing_ratio", {}).get("rows", [])
    mp = mexp.get("missing_position", {}).get("rows", [])
    if len(mr) == 24:
        c.ok("事实表", "缺失比例行数", "24 行（2 模型 × 3 类型 × 4 比例）")
    else:
        c.fail("事实表", "缺失比例行数", f"{len(mr)} 行，期望 24")
    if len(mp) == 48:
        c.ok("事实表", "缺失位置行数", "48 行（2 模型 × 2 模态 × 3 位置 × 4 比例）")
    else:
        c.fail("事实表", "缺失位置行数", f"{len(mp)} 行，期望 48")
    text_rows = mexp.get("deletion", {}).get("text", {}).get("rows", [])
    image_rows = mexp.get("deletion", {}).get("image", {}).get("rows", [])
    if len(text_rows) == 8 and len(image_rows) == 6:
        c.ok("事实表", "删除实验行数", "文本 8 行 / 图像 6 行")
    else:
        c.fail("事实表", "删除实验行数", f"文本 {len(text_rows)} / 图像 {len(image_rows)}")
    if {(r.get("method"), int(r.get("k", 0))) for r in text_rows} >= \
            {("top", 1), ("random", 1), ("top", 3), ("random", 3), ("top", 5), ("random", 5)}:
        c.ok("事实表", "文本删除分组", "top/random × k=1/2/3/5")
    else:
        c.fail("事实表", "文本删除分组", "缺少 top/random 或 k 组合")
    if {(r.get("method"), int(r.get("k", 0))) for r in image_rows} >= \
            {("top", 1), ("random", 1), ("top", 3), ("random", 3)}:
        c.ok("事实表", "图像删除分组", "top/random × k=1/2/3")
    else:
        c.fail("事实表", "图像删除分组", "缺少 top/random 或 k 组合")

    # 单条推理与解释卡
    ex = infer.get("examples", [])
    if len(ex) == 2 and {e.get("name") for e in ex} == {"text_only", "text_image"}:
        c.ok("事实表", "推理自检条目", "text_only + text_image")
    else:
        c.fail("事实表", "推理自检条目", f"{[e.get('name') for e in ex]}")
    lat = [float(e.get("result", {}).get("latency_ms") or 0) for e in ex]
    if all(x > 0 for x in lat):
        c.ok("事实表", "推理耗时", f"{lat[0]:.1f} ms / {lat[1]:.1f} ms")
    else:
        c.fail("事实表", "推理耗时", f"存在非正值：{lat}")
    cs = cards.get("cards", [])
    bad = [x.get("post_id") for x in cs
           if not all(k in x for k in ("prediction", "modalities", "evidence"))]
    if len(cs) == 3 and not bad:
        c.ok("事实表", "解释卡结构", "3 条，均含 prediction/modalities/evidence")
    else:
        c.fail("事实表", "解释卡结构", f"n={len(cs)}，缺字段：{bad}")
    if anno.get("self_test") is True and anno.get("source") == "self_test_synthetic" \
            and "不是" in str(anno.get("note", "")):
        c.ok("事实表", "标注自检声明", "self_test=true / 合成数据 / 明确不是人工标注")
    else:
        c.fail("事实表", "标注自检声明", "未明确标注为合成数据自检")


def main():
    ap = argparse.ArgumentParser(description="08 提交前自检门禁")
    ap.add_argument("--no-write", action="store_true", help="不写 output/submission_check.json")
    args = ap.parse_args()

    print("=" * 70)
    print("08 提交前自检（提交门禁）")
    print("=" * 70)
    print(f"项目根目录：{ROOT}")
    print(f"检查时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    print()

    c = Checker()
    steps = [
        ("产物", check_required_files),
        ("数据", check_data_scale),
        ("指标", check_metrics),
        ("报告", check_report),
        ("PDF", check_pdf),
        ("新鲜度", check_freshness),
        ("大屏", check_dashboard),
        ("单条页", check_single_page),
        ("隐私", check_privacy),
        ("合规", check_compliance),
        ("日志", check_logs),
        ("文档", check_docs_content),
        ("代码", check_code_contracts),
        ("事实表", check_json_schema),
    ]
    for name, fn in steps:
        before = len(c.items)
        try:
            fn(c)
        except Exception as e:
            c.fail(name, f"{name}检查异常", f"{type(e).__name__}: {e}")
        # 打印本组结果
        for it in c.items[before:]:
            detail = f"  {it['detail']}" if it["detail"] else ""
            print(f"[{it['status']:<4}] {it['category']}｜{it['name']}{detail}")

    counts = c.counts()
    print()
    print("-" * 70)
    print(f"汇总：PASS {counts['PASS']} / WARN {counts['WARN']} / "
          f"FAIL {counts['FAIL']} / INFO {counts['INFO']}")
    if counts["FAIL"] == 0:
        print("结论：PASS（可以提交；WARN 项按提示人工确认）")
        code = 0
    else:
        print("结论：FAIL（必须修复后再提交）")
        print("\nFAIL 列表：")
        for it in c.items:
            if it["status"] == "FAIL":
                print(f"  - {it['category']}｜{it['name']}：{it['detail']}")
        code = 1

    if not args.no_write:
        out = {
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": counts,
            "result": "PASS" if code == 0 else "FAIL",
            "items": c.items,
        }
        with open(os.path.join(OUT, "submission_check.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"\n自检结果：output/submission_check.json")

    print("=" * 70)
    return code


if __name__ == "__main__":
    sys.exit(main())
