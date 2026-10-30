# -*- coding: utf-8 -*-
"""13 标注一致性工具：Cohen's Kappa、分歧清单与标注模板。

用途
----
- 对双人独立标注 CSV 计算极性 Kappa、强度 Kappa（二次加权）、MAE、Pearson；
- 输出分歧条目，供第三人裁决；
- 生成空白标注模板（只含候选样本与待填字段，不伪造人工标签）；
- `--self-test` 用合成数据验证计算链路，结果写入
  output/annotation_self_test.json，并明确标注“工具自检，不是人工标注结果”。

输入 CSV 列（大小写不敏感，允许中文列名）
----------------------------------------
item_id, event, post_type, text,
annotator_a, annotator_b, intensity_a, intensity_b,
adjudicated_label, adjudicated_intensity, notes

极性取值：-1/0/1 或 负面/中性/正面；
强度取值：-3~3 的整数（可选）。

边界
----
本工具只负责“算一致性”，不替代人工标注；项目当前没有完成方案要求的
1,500 条双人标注，因此文档中必须如实写“工具已实现、标注未执行”。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import OUT, ROOT, now_str, write_json

LABEL_MAP = {
    "-1": -1, "0": 0, "1": 1,
    "负面": -1, "负向": -1, "negative": -1, "neg": -1,
    "中性": 0, "neutral": 0, "neu": 0,
    "正面": 1, "正向": 1, "positive": 1, "pos": 1,
}
LABEL_NAME = {-1: "负面", 0: "中性", 1: "正面"}


def norm_label(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    return LABEL_MAP.get(s.lower(), LABEL_MAP.get(s))


def norm_intensity(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        x = int(round(float(s)))
    except ValueError:
        return None
    return x if -3 <= x <= 3 else None


def cohen_kappa(y1, y2) -> dict:
    n = len(y1)
    if n == 0:
        return {"n": 0, "kappa": None, "observed_agreement": None,
                "expected_agreement": None}
    po = sum(a == b for a, b in zip(y1, y2)) / n
    ca, cb = Counter(y1), Counter(y2)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    kappa = None if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)
    return {
        "n": n,
        "kappa": None if kappa is None else round(float(kappa), 6),
        "observed_agreement": round(float(po), 6),
        "expected_agreement": round(float(pe), 6),
    }


def confusion(y1, y2) -> dict:
    out = {}
    for a, b in zip(y1, y2):
        out.setdefault(str(a), Counter())[str(b)] += 1
    return {k: dict(v) for k, v in out.items()}


def read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            low = {str(k).strip().lower(): v for k, v in r.items()}
            rows.append({
                "item_id": low.get("item_id") or low.get("id") or "",
                "event": low.get("event") or low.get("事件") or "",
                "post_type": low.get("post_type") or low.get("帖子类型") or "",
                "text": low.get("text") or low.get("正文") or "",
                "annotator_a": norm_label(low.get("annotator_a") or low.get("标注员a")),
                "annotator_b": norm_label(low.get("annotator_b") or low.get("标注员b")),
                "intensity_a": norm_intensity(low.get("intensity_a") or low.get("强度a")),
                "intensity_b": norm_intensity(low.get("intensity_b") or low.get("强度b")),
                "notes": low.get("notes") or low.get("备注") or "",
            })
    return rows


def analyze(rows: list[dict], source: str) -> dict:
    valid = [r for r in rows if r["annotator_a"] is not None and r["annotator_b"] is not None]
    ya = [r["annotator_a"] for r in valid]
    yb = [r["annotator_b"] for r in valid]
    pol = cohen_kappa(ya, yb)
    pol["confusion"] = confusion(ya, yb)
    pol["per_class"] = {}
    for lab in (-1, 0, 1):
        idx = [i for i, v in enumerate(ya) if v == lab]
        if idx:
            pol["per_class"][LABEL_NAME[lab]] = {
                "n": len(idx),
                "agreement": round(float(np.mean([yb[i] == lab for i in idx])), 6),
            }
    disagreements = [{
        "item_id": r["item_id"], "event": r["event"], "post_type": r["post_type"],
        "text": r["text"][:120],
        "annotator_a": LABEL_NAME[r["annotator_a"]],
        "annotator_b": LABEL_NAME[r["annotator_b"]],
        "intensity_a": r["intensity_a"], "intensity_b": r["intensity_b"],
        "notes": r["notes"],
    } for r in valid if r["annotator_a"] != r["annotator_b"]]

    inten = [r for r in valid if r["intensity_a"] is not None and r["intensity_b"] is not None]
    intensity = {"n": len(inten)}
    if inten:
        ia = [r["intensity_a"] for r in inten]
        ib = [r["intensity_b"] for r in inten]
        intensity.update({
            "kappa": round(float(cohen_kappa_score(ia, ib)), 6),
            "quadratic_weighted_kappa": round(float(cohen_kappa_score(
                ia, ib, weights="quadratic", labels=list(range(-3, 4)))), 6),
            "mae": round(float(np.mean(np.abs(np.array(ia) - np.array(ib)))), 6),
            "pearson": round(float(np.corrcoef(ia, ib)[0, 1]), 6)
            if len(set(ia)) > 1 and len(set(ib)) > 1 else None,
        })

    by_event = {}
    for r in valid:
        key = r["event"] or "未填写"
        by_event.setdefault(key, {"a": [], "b": []})
        by_event[key]["a"].append(r["annotator_a"])
        by_event[key]["b"].append(r["annotator_b"])
    by_event = {k: cohen_kappa(v["a"], v["b"]) for k, v in by_event.items()}

    by_type = {}
    for r in valid:
        key = r["post_type"] or "未填写"
        by_type.setdefault(key, {"a": [], "b": []})
        by_type[key]["a"].append(r["annotator_a"])
        by_type[key]["b"].append(r["annotator_b"])
    by_type = {k: cohen_kappa(v["a"], v["b"]) for k, v in by_type.items()}

    kappa = pol.get("kappa")
    if kappa is None:
        conclusion = "样本不足，无法判断"
    elif kappa >= 0.8:
        conclusion = "几乎完全一致（≥0.8）"
    elif kappa >= 0.6:
        conclusion = "达到方案门槛（≥0.6）"
    elif kappa >= 0.4:
        conclusion = "中等一致（0.4~0.6），需修订标注规范"
    else:
        conclusion = "一致性偏低（<0.4），需重新培训/预试"

    return {
        "generated_at": now_str(),
        "source": source,
        "n_rows": len(rows),
        "n_valid": len(valid),
        "n_disagreements": len(disagreements),
        "polarity": pol,
        "intensity": intensity,
        "by_event": by_event,
        "by_post_type": by_type,
        "disagreements": disagreements,
        "conclusion": conclusion,
        "note": "本结果仅针对输入 CSV；若为 --self-test，则为合成数据，不是人工标注结果。",
    }


def self_test() -> dict:
    rng = np.random.default_rng(42)
    rows = []
    labels = [-1, 0, 1]
    for i in range(100):
        a = int(rng.choice(labels, p=[0.35, 0.3, 0.35]))
        b = a if rng.random() < 0.86 else int(rng.choice([x for x in labels if x != a]))
        ia = int(np.clip(a * 2 + rng.integers(-1, 2), -3, 3))
        ib = int(np.clip(ia + (0 if rng.random() < 0.8 else rng.integers(-1, 2)), -3, 3))
        rows.append({
            "item_id": f"selftest_{i:03d}",
            "event": f"事件{(i % 3) + 1}",
            "post_type": ["纯文本", "图文", "视频"][i % 3],
            "text": "合成自检样本，不对应真实微博。",
            "annotator_a": a, "annotator_b": b,
            "intensity_a": ia, "intensity_b": ib,
            "notes": "",
        })
    result = analyze(rows, "self_test_synthetic")
    result["self_test"] = True
    result["note"] = ("工具自检使用合成标注，只证明 Kappa/加权 Kappa/分歧统计链路可运行；"
                      "不是本项目的人工标注结果，不能用于宣称已完成 1,500 条标注。")
    return result


def make_template(path: str, n: int = 30, force: bool = False) -> str:
    if os.path.exists(path) and not force:
        print(f"[INFO] 模板已存在，跳过：{os.path.relpath(path, ROOT)}（如需覆盖请加 --force）")
        return path
    src = os.path.join(ROOT, "data", "crawl", "weibo_1000.csv")
    if not os.path.exists(src):
        raise SystemExit("缺少 data/crawl/weibo_1000.csv，无法生成候选模板")
    rows = []
    with open(src, encoding="utf-8-sig", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= n:
                break
            rows.append({
                "item_id": r.get("post_id", ""),
                "event": "待填写事件",
                "post_type": "图文",
                "text": r.get("text", ""),
                "annotator_a": "",
                "annotator_b": "",
                "intensity_a": "",
                "intensity_b": "",
                "adjudicated_label": "",
                "adjudicated_intensity": "",
                "notes": "空白模板：请双人独立填写，分歧交第三人裁决",
            })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    ap = argparse.ArgumentParser(description="双人标注 Cohen's Kappa 与分歧统计")
    ap.add_argument("--input", help="双人标注 CSV")
    ap.add_argument("--output", default=os.path.join(OUT, "annotation_kappa.json"))
    ap.add_argument("--self-test", action="store_true", help="合成数据自检")
    ap.add_argument("--make-template", action="store_true", help="生成空白标注模板")
    ap.add_argument("--template-out",
                    default=os.path.join(ROOT, "data", "annotation", "annotation_template.csv"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.make_template:
        path = make_template(args.template_out, force=args.force)
        print(f"[OK] 空白标注模板 -> {os.path.relpath(path, ROOT)}")
        print("     说明：模板只含候选样本与空字段，不包含任何人工标签。")
        if not args.self_test and not args.input:
            return

    if args.self_test:
        result = self_test()
        write_json(os.path.join(OUT, "annotation_self_test.json"), result)
        print("[OK] 标注一致性工具自检（合成数据）")
        print(f"  极性 Kappa={result['polarity']['kappa']}，"
              f"一致率={result['polarity']['observed_agreement']}，"
              f"分歧={result['n_disagreements']} 条")
        print(f"  强度二次加权 Kappa={result['intensity'].get('quadratic_weighted_kappa')}，"
              f"MAE={result['intensity'].get('mae')}")
        print("  结果 -> output/annotation_self_test.json")
        print("  注意：这是工具自检，不是人工标注结果。")
        if not args.input:
            return

    if not args.input:
        raise SystemExit("请提供 --input，或使用 --self-test / --make-template")
    rows = read_rows(args.input)
    result = analyze(rows, args.input)
    write_json(args.output, result)
    print(f"[OK] {result['n_valid']}/{result['n_rows']} 条有效标注 -> "
          f"{os.path.relpath(args.output, ROOT)}")
    print(f"  极性 Kappa={result['polarity']['kappa']}，"
          f"一致率={result['polarity']['observed_agreement']}，"
          f"分歧={result['n_disagreements']} 条")
    print(f"  结论：{result['conclusion']}")


if __name__ == "__main__":
    main()
