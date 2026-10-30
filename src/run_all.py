# -*- coding: utf-8 -*-
"""run_all 一键流水线：公开数据导入 → 校验 → 采集自检 → 落盘 → 预处理
→ 分析 → 建模 → 缺失/删除实验 → 单条推理 → 解释卡 → 大屏数据
→ 报告图 → 标注自检 → 报告填充。

设计原则
--------
- 不使用演示参数（demo），不生成合成训练数据；
- 每步失败即停，日志写入 output/logs/<step>.log（UTF-8）；
- 支持 --from 从指定步骤续跑；
- 不自动导出 PDF、不自动提交，避免“文档未核对就打包”。

用法
----
    python src/run_all.py
    python src/run_all.py --from 04
    python src/run_all.py --list
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOGS = os.path.join(ROOT, "output", "logs")

# (步骤 id, 显示名, 脚本, 参数, 必须存在的产物)
STEPS = [
    ("01a", "公开数据集导入", "01a_import_scidb.py", [],
     ["data/raw/posts_raw.csv", "data/raw/import_manifest.json"]),
    ("01", "数据质量校验", "01_data_collection.py", ["--check"],
     ["output/data_quality.json"]),
    ("01b", "合规采集器自检", "01b_collect_public.py", ["--self-test"],
     ["output/crawler_self_test.json"]),
    ("01c", "1,000 条落盘样本", "01c_export_crawl_1000.py", ["--n", "1000"],
     ["data/crawl/weibo_1000.csv", "data/crawl/manifest.json"]),
    ("02", "预处理与图像特征", "02_preprocess.py", [],
     ["data/processed/posts_full.csv", "data/processed/posts_clean.csv",
      "data/processed/image_features.npy"]),
    ("03", "舆情分析", "03_analysis.py", [],
     ["output/analysis_results.json"]),
    ("04", "多模态建模", "04_modeling.py", [],
     ["output/models_metrics.json", "output/explainability.json"]),
    ("12", "缺失比例/位置与删除实验", "12_missing_explain_experiments.py", [],
     ["output/missing_experiments.json",
      "output/tables/缺失比例曲线.csv", "output/tables/缺失位置曲线.csv",
      "output/tables/删除实验.csv",
      "docs/figures/fig11_missing_ratio.png", "docs/figures/fig12_deletion.png"]),
    ("10", "单条推理自检", "10_inference_service.py", ["--self-test"],
     ["output/inference_self_test.json"]),
    ("11", "解释卡与单条页", "11_export_explanation_cards.py", [],
     ["output/explanation_cards.json", "dashboard/single.html"]),
    ("05", "大屏数据", "05_make_dashboard_data.py", [],
     ["dashboard/data.js"]),
    ("06", "报告插图与大屏截图", "06_make_report_figures.py", [],
     ["docs/figures/fig1_architecture.png", "docs/figures/fig10_external.png"]),
    ("13", "标注一致性工具自检", "13_annotation_kappa.py",
     ["--self-test", "--make-template"],
     ["output/annotation_self_test.json", "data/annotation/annotation_template.csv"]),
    ("09", "报告与实验记录", "09_fill_report.py", [],
     ["docs/01_研究报告.md", "docs/04_实验记录.md"]),
]


def run_step(step):
    sid, name, script, args, outputs = step
    script_path = os.path.join(HERE, script)
    if not os.path.exists(script_path):
        raise SystemExit(f"[FAIL] 缺少脚本：{script_path}")

    os.makedirs(LOGS, exist_ok=True)
    log_path = os.path.join(LOGS, f"{sid}_{os.path.splitext(script)[0]}.log")
    cmd = [sys.executable, script_path, *args]
    print("\n" + "=" * 70)
    print(f"[{sid}] {name}  ({script})")
    print("=" * 70)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(out)
    # 控制台同步输出（避免一次性刷屏，保留尾部）
    if out:
        print(out.rstrip())
    if proc.returncode != 0:
        print(f"\n[FAIL] {sid} 失败，退出码 {proc.returncode}；完整日志：{log_path}")
        raise SystemExit(proc.returncode)
    missing = [p for p in outputs if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        print(f"\n[FAIL] {sid} 完成但缺少产物：{missing}")
        raise SystemExit(1)
    print(f"\n[OK] {sid} 完成，耗时 {elapsed:.1f}s；日志 {os.path.relpath(log_path, ROOT)}")


def main():
    ap = argparse.ArgumentParser(description="微博多模态舆情一键流水线")
    ap.add_argument("--from", dest="start", default=None,
                    help="从指定步骤 id 开始，如 04")
    ap.add_argument("--list", action="store_true", help="列出步骤后退出")
    args = ap.parse_args()

    if args.list:
        for sid, name, script, _, _ in STEPS:
            print(f"{sid:>4}  {name:<20}  src/{script}")
        return

    ids = [s[0] for s in STEPS]
    start_idx = 0
    if args.start:
        if args.start not in ids:
            raise SystemExit(f"--from 必须是 {ids} 之一")
        start_idx = ids.index(args.start)

    t0 = time.time()
    print("=" * 70)
    print("微博多模态舆情 · 一键流水线（不使用演示参数）")
    print("=" * 70)
    print(f"Python：{sys.version.split()[0]}")
    print(f"工作目录：{ROOT}")
    print(f"执行步骤：{' → '.join(ids[start_idx:])}")

    for step in STEPS[start_idx:]:
        run_step(step)

    total = time.time() - t0
    print("\n" + "=" * 70)
    print(f"[全部完成] 总耗时 {total / 60:.1f} 分钟")
    print("=" * 70)
    print("下一步：")
    print("  1) python src/07_export_docs.py --pdf")
    print("  2) python src/08_submission_check.py   # 必须 FAIL 0")
    print("\n提示：如果只改了文档，不必重跑本流水线；改数据/模型后必须重跑 "
          "03/04 → 12 → 10/11 → 05/06 → 13 → 09。")


if __name__ == "__main__":
    main()
