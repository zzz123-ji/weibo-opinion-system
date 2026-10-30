# -*- coding: utf-8 -*-
"""09 用事实表填充研究报告模板，并生成实验记录。

输入
----
docs/templates/01_研究报告.template.md
output/analysis_results.json
output/models_metrics.json
output/explainability.json
data/crawl/manifest.json

输出
----
docs/01_研究报告.md
docs/04_实验记录.md
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import (DOCS, LABEL_NAMES, OUT, ROOT, TEMPLATES, now_str,
                        read_json)

TEMPLATE = os.path.join(TEMPLATES, "01_研究报告.template.md")
REPORT = os.path.join(DOCS, "01_研究报告.md")
EXP_LOG = os.path.join(DOCS, "04_实验记录.md")


def pct(x, nd=2):
    return f"{float(x) * 100:.{nd}f}%"


def f4(x):
    return f"{float(x):.4f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main():
    ana = read_json(os.path.join(OUT, "analysis_results.json"))
    met = read_json(os.path.join(OUT, "models_metrics.json"))
    exp = read_json(os.path.join(OUT, "explainability.json"))
    crawl = read_json(os.path.join(ROOT, "data", "crawl", "manifest.json"), {})
    missing = read_json(os.path.join(OUT, "missing_experiments.json"), {})
    inference = read_json(os.path.join(OUT, "inference_self_test.json"), {})
    cards = read_json(os.path.join(OUT, "explanation_cards.json"), {})
    if not (ana and met and exp):
        raise SystemExit("缺少 analysis_results.json / models_metrics.json / explainability.json")

    missing_ratio = missing.get("missing_ratio") or met.get("missing_ratio") or {}
    missing_position = missing.get("missing_position") or met.get("missing_position") or {}
    deletion = missing.get("deletion") or met.get("deletion") or {}
    if not missing_ratio.get("rows"):
        raise SystemExit("缺少缺失比例实验，请先运行 12_missing_explain_experiments.py")
    if not deletion.get("text", {}).get("rows"):
        raise SystemExit("缺少删除实验，请先运行 12_missing_explain_experiments.py")

    labels = ana["data"]["label_counts"]
    ratios = ana["data"]["label_ratio"]
    trend = ana["trend"]
    topics = ana["topics"]["list"]
    provinces = ana["provinces"]
    hashtags = ana["hashtags"]
    alerts = ana["alerts"]
    complete = met["complete"]
    robust = met["robustness"]
    ext = met["external_no_image"]
    gates = met["gates"]

    def get_rob(model, sc):
        return next(r for r in robust if r["model"] == model and r["scenario"] == sc)

    def get_miss(model, modality, ratio):
        return next(r for r in missing_ratio["rows"]
                    if r["model"] == model and r["modality"] == modality
                    and abs(float(r["ratio"]) - ratio) < 1e-9)

    def get_del(modality, method, k):
        return next(r for r in deletion[modality]["rows"]
                    if r["method"] == method and int(r["k"]) == k)

    def get_pos(modality, position, ratio):
        return next(r for r in missing_position["rows"]
                    if r["model"] == "gated" and r["modality"] == modality
                    and r["position"] == position
                    and abs(float(r["ratio"]) - ratio) < 1e-9)

    def get_infer(name):
        for e in inference.get("examples", []):
            if e.get("name") == name:
                return e.get("result", {}).get("latency_ms")
        return None

    top_topic = topics[0]
    neg_topics = [t for t in topics if t["n"] >= 500]
    neg_topic = max(neg_topics, key=lambda x: x["neg_ratio"])
    top_alert = alerts[0] if alerts else {"date": "—", "n": 0, "neg_ratio": 0, "reason": ""}
    top_prov = provinces[0]
    top_tag = hashtags[0]
    gate_all = next(g for g in gates if g["group"] == "all")
    gate_by = {g["label"]: g for g in gates if g["group"] == "true_label"}

    # 混淆矩阵
    cm = complete["gated"]["confusion_matrix"]
    # labels order -1,0,1
    neg_recall = cm[0][0] / max(sum(cm[0]), 1)
    neu_recall = cm[1][1] / max(sum(cm[1]), 1)
    pos_recall = cm[2][2] / max(sum(cm[2]), 1)

    # 图像遮挡
    img_grid = exp["image_occlusion"]["grid"]
    img_center = img_grid[1][1]
    img_tl = img_grid[0][0]
    img_br = img_grid[2][2]

    vals = {
        "total_rows": f"{ana['data']['total_rows']:,}",
        "sample_rows": f"{ana['data']['sample_rows']:,}",
        "external_n": f"{met['data']['external_no_image']:,}",
        "crawl_n": f"{crawl.get('n', 0):,}",
        "crawl_images": f"{crawl.get('images', 0):,}",
        "crawl_mb": f"{crawl.get('image_total_mb', 0):.2f}",
        "crawl_users": f"{crawl.get('unique_user_id', 0):,}",
        "crawl_date_min": crawl.get("date_min", ""),
        "crawl_date_max": crawl.get("date_max", ""),
        "neg_count": f"{labels['-1']:,}",
        "neu_count": f"{labels['0']:,}",
        "pos_count": f"{labels['1']:,}",
        "neg_pct": pct(ratios["-1"]),
        "neu_pct": pct(ratios["0"]),
        "pos_pct": pct(ratios["1"]),
        "train_n": f"{met['data']['train']:,}",
        "val_n": f"{met['data']['val']:,}",
        "test_n": f"{met['data']['test']:,}",
        "date_rows": f"{ana['data']['date_rows']:,}",
        "date_cov": pct(ana["data"]["date_coverage"]),
        "date_min": ana["data"]["date_min"],
        "date_max": ana["data"]["date_max"],
        "n_days": ana["data"]["unique_days"],
        "pre_mean": f"{trend['pre_jan20_mean_n']:.1f}",
        "post_mean": f"{trend['post_jan20_mean_n']:.1f}",
        "growth": pct(trend["volume_growth"], 1),
        "peak_date": trend["peak_volume_date"],
        "peak_n": f"{trend['peak_volume_n']:,}",
        "peak_neg_date": trend["peak_neg_date"],
        "peak_neg_pct": pct(trend["peak_neg_ratio"]),
        "baseline_neg_pct": pct(trend["baseline_neg_ratio"]),
        "overall_neg": pct(trend["overall_neg_ratio"]),
        "n_alerts": len(alerts),
        "top_alert_date": top_alert["date"],
        "top_alert_n": f"{top_alert['n']:,}",
        "top_alert_neg": pct(top_alert["neg_ratio"]),
        "top_alert_reason": top_alert["reason"],
        "top_topic": top_topic["label"],
        "top_topic_n": f"{top_topic['n']:,}",
        "top_topic_share": pct(top_topic["share"]),
        "top_topic_neg": pct(top_topic["neg_ratio"]),
        "neg_topic": neg_topic["label"],
        "neg_topic_neg": pct(neg_topic["neg_ratio"]),
        "neg_keywords": "、".join(x["word"] for x in ana["keywords"]["-1"][:8]),
        "pos_keywords": "、".join(x["word"] for x in ana["keywords"]["1"][:8]),
        "province_rows": f"{ana['data']['province_rows']:,}",
        "province_cov": pct(ana["data"]["province_coverage"]),
        "hubei_n": f"{top_prov['n']:,}",
        "hubei_share": pct(top_prov["share"]),
        "hubei_neg": pct(top_prov["neg_ratio"]),
        "top_tag": top_tag["tag"],
        "top_tag_n": f"{top_tag['n']:,}",
        "top_tag_pos": pct(top_tag["pos_ratio"]),
        "top_tag_neg": pct(top_tag["neg_ratio"]),
        "svd_var": f"{met['data']['text_svd_explained_variance']:.3f}",
        "gated_acc": f4(complete["gated"]["accuracy"]),
        "gated_f1": f4(complete["gated"]["macro_f1"]),
        "gated_ci_low": f4(complete["gated"]["bootstrap_macro_f1"]["low"]),
        "gated_ci_high": f4(complete["gated"]["bootstrap_macro_f1"]["high"]),
        "text_f1": f4(complete["text_lr"]["macro_f1"]),
        "image_f1": f4(complete["image_lr"]["macro_f1"]),
        "early_f1": f4(complete["early_lr"]["macro_f1"]),
        "late_f1": f4(complete["late_lr"]["macro_f1"]),
        "early_mlp_f1": f4(complete["early_mlp"]["macro_f1"]),
        "gated_equal_f1": f4(complete["gated_equal"]["macro_f1"]),
        "gated_minus_text": f"{complete['gated']['macro_f1'] - complete['text_lr']['macro_f1']:+.4f}",
        "gate_gain": f"{complete['gated']['macro_f1'] - complete['gated_equal']['macro_f1']:+.4f}",
        "gated_vs_mlp": f"{complete['gated']['macro_f1'] - complete['early_mlp']['macro_f1']:+.4f}",
        "gated_text_f1": f4(get_rob("gated", "text-only")["macro_f1"]),
        "gated_image_f1": f4(get_rob("gated", "image-only")["macro_f1"]),
        "gated_both_f1": f4(get_rob("gated", "both-missing")["macro_f1"]),
        "early_text_f1": f4(get_rob("early_lr", "text-only")["macro_f1"]),
        "early_image_f1": f4(get_rob("early_lr", "image-only")["macro_f1"]),
        "late_text_f1": f4(get_rob("late_lr", "text-only")["macro_f1"]),
        "late_image_f1": f4(get_rob("late_lr", "image-only")["macro_f1"]),
        "early_mlp_text_f1": f4(get_rob("early_mlp", "text-only")["macro_f1"]),
        "early_mlp_image_f1": f4(get_rob("early_mlp", "image-only")["macro_f1"]),
        "ext_gated_f1": f4(ext["gated"]["macro_f1"]),
        "ext_text_f1": f4(ext["text_lr"]["macro_f1"]),
        "ext_early_f1": f4(ext["early_lr"]["macro_f1"]),
        "ext_late_f1": f4(ext["late_lr"]["macro_f1"]),
        "ext_early_mlp_f1": f4(ext["early_mlp"]["macro_f1"]),
        "early_drop": f"{complete['early_lr']['macro_f1'] - ext['early_lr']['macro_f1']:.4f}",
        "gate_text_pct": pct(gate_all["text_gate"], 1),
        "gate_image_pct": pct(gate_all["image_gate"], 1),
        "gate_neg_text": pct(gate_by["负面"]["text_gate"], 1),
        "gate_neg_image": pct(gate_by["负面"]["image_gate"], 1),
        "gate_neu_text": pct(gate_by["中性"]["text_gate"], 1),
        "gate_neu_image": pct(gate_by["中性"]["image_gate"], 1),
        "gate_pos_text": pct(gate_by["正面"]["text_gate"], 1),
        "gate_pos_image": pct(gate_by["正面"]["image_gate"], 1),
        "text_occ_n": exp["text_occlusion"].get("n_samples", 0),
        "text_occ_pert": exp["text_occlusion"].get("n_perturbations", 0),
        "image_occ_n": exp["image_occlusion"].get("n_samples", 0),
        "img_center_drop": f"{img_center:.4f}",
        "img_tl_drop": f"{img_tl:.4f}",
        "img_br_drop": f"{img_br:.4f}",
        "miss_text_10": f4(get_miss("gated", "text", 0.1)["macro_f1_mean"]),
        "miss_text_70": f4(get_miss("gated", "text", 0.7)["macro_f1_mean"]),
        "miss_text_drop": f"{get_miss('gated', 'text', 0.1)['macro_f1_mean'] - get_miss('gated', 'text', 0.7)['macro_f1_mean']:.4f}",
        "miss_both_10": f4(get_miss("gated", "both", 0.1)["macro_f1_mean"]),
        "miss_both_70": f4(get_miss("gated", "both", 0.7)["macro_f1_mean"]),
        "miss_both_drop": f"{get_miss('gated', 'both', 0.1)['macro_f1_mean'] - get_miss('gated', 'both', 0.7)['macro_f1_mean']:.4f}",
        "miss_image_min": f4(min(get_miss("gated", "image", r)["macro_f1_mean"] for r in (0.1, 0.3, 0.5, 0.7))),
        "miss_image_max": f4(max(get_miss("gated", "image", r)["macro_f1_mean"] for r in (0.1, 0.3, 0.5, 0.7))),
        "del_text_k1_top": f4(get_del("text", "top", 1)["mean_prob_drop"]),
        "del_text_k1_rand": f4(get_del("text", "random", 1)["mean_prob_drop"]),
        "del_text_k3_top": f4(get_del("text", "top", 3)["mean_prob_drop"]),
        "del_text_k3_rand": f4(get_del("text", "random", 3)["mean_prob_drop"]),
        "del_text_k5_top": f4(get_del("text", "top", 5)["mean_prob_drop"]),
        "del_text_k5_rand": f4(get_del("text", "random", 5)["mean_prob_drop"]),
        "del_img_k1_top": f4(get_del("image", "top", 1)["mean_prob_drop"]),
        "del_img_k1_rand": f4(get_del("image", "random", 1)["mean_prob_drop"]),
        "del_img_k3_top": f4(get_del("image", "top", 3)["mean_prob_drop"]),
        "del_img_k3_rand": f4(get_del("image", "random", 3)["mean_prob_drop"]),
        "del_text_n": f"{int(deletion['text'].get('n_candidates', 0)):,}",
        "del_img_n": f"{int(deletion['image'].get('n_candidates', 0)):,}",
        "infer_text_ms": f"{float(get_infer('text_only') or 0):.1f}",
        "infer_text_image_ms": f"{float(get_infer('text_image') or 0):.1f}",
        "cards_n": f"{int(cards.get('n', 0))}",
        "neg_recall": pct(neg_recall, 1),
        "neu_recall": pct(neu_recall, 1),
        "pos_recall": pct(pos_recall, 1),
        "neg_to_neu": pct(cm[0][1] / max(sum(cm[0]), 1), 1),
        "neu_to_neg": pct(cm[1][0] / max(sum(cm[1]), 1), 1),
        "neu_to_pos": pct(cm[1][2] / max(sum(cm[1]), 1), 1),
        "pos_to_neu": pct(cm[2][1] / max(sum(cm[2]), 1), 1),
    }

    # ---------- 表 ----------
    topic_table = md_table(
        ["排名", "话题（top terms）", "帖子数", "占比", "负向率", "正向率"],
        [[i + 1, f"**{t['label']}**（{' / '.join(t['top_terms'][:5])}）",
          f"{t['n']:,}", pct(t["share"]), pct(t["neg_ratio"]), pct(t["pos_ratio"])]
         for i, t in enumerate(topics)])
    labels_model = {
        "text_lr": "文本单模态（TF-IDF+SVD+LR）", "image_lr": "图像单模态（ResNet+统计+LR）",
        "early_lr": "早期融合（特征拼接+LR）", "late_lr": "晚期融合（概率加权）",
        "gated": "门控融合（本文）", "gated_equal": "门控融合-等权消融",
        "early_mlp": "早期融合 MLP（无门控对照）",
    }
    model_table = md_table(
        ["模型", "Accuracy", "Macro-F1", "95% CI", "Weighted-F1"],
        [[labels_model[k], f4(complete[k]["accuracy"]), f4(complete[k]["macro_f1"]),
          f"{complete[k]['bootstrap_macro_f1']['low']:.4f}~{complete[k]['bootstrap_macro_f1']['high']:.4f}",
          f4(complete[k]["weighted_f1"])]
         for k in ["text_lr", "image_lr", "early_lr", "late_lr", "gated", "gated_equal", "early_mlp"]])
    scen_name = {"text-only": "仅文本", "image-only": "仅图像", "both-missing": "双缺失"}
    robust_table = md_table(
        ["模型", "场景", "Accuracy", "Macro-F1"],
        [[labels_model[r["model"]], scen_name[r["scenario"]],
          f4(r["accuracy"]), f4(r["macro_f1"])]
         for r in robust if r["model"] in ("early_lr", "late_lr", "gated", "early_mlp")])
    ext_table = md_table(
        ["模型", "Accuracy", "Macro-F1", "95% CI"],
        [[labels_model[k], f4(ext[k]["accuracy"]), f4(ext[k]["macro_f1"]),
          f"{ext[k]['bootstrap_macro_f1']['low']:.4f}~{ext[k]['bootstrap_macro_f1']['high']:.4f}"]
         for k in ["text_lr", "early_lr", "late_lr", "gated", "early_mlp"]])
    province_table = md_table(
        ["排名", "省份", "提及帖子数", "占全量比例", "负向率", "正向率"],
        [[i + 1, p["province"], f"{p['n']:,}", pct(p["share"]), pct(p["neg_ratio"]), pct(p["pos_ratio"])]
         for i, p in enumerate(provinces[:10])])
    hashtag_table = md_table(
        ["排名", "话题标签", "帖子数", "负向率", "正向率"],
        [[i + 1, f"#{h['tag']}#", f"{h['n']:,}", pct(h["neg_ratio"]), pct(h["pos_ratio"])]
         for i, h in enumerate(hashtags[:10])])
    alert_table = md_table(
        ["日期", "声量", "负向率", "z(负向)", "z(声量)", "等级", "触发原因"],
        [[a["date"], f"{a['n']:,}", pct(a["neg_ratio"]), f"{a['z_neg']:.2f}",
          f"{a['z_volume']:.2f}", a["level"], a["reason"]] for a in alerts])
    gate_table = md_table(
        ["分组", "类别", "样本数", "文本门控", "图像门控"],
        [[g["group"], g["label"], f"{g['n']:,}", pct(g["text_gate"], 1), pct(g["image_gate"], 1)]
         for g in gates])
    text_occ_table = md_table(
        ["排名", "token", "出现次数", "平均概率下降", "最大概率下降"],
        [[i + 1, t["token"], t["n"], f"{t['mean_prob_drop']:.4f}", f"{t['max_prob_drop']:.4f}"]
         for i, t in enumerate(exp["text_occlusion"].get("top_tokens", [])[:8])])
    miss_model_name = {"gated": "门控融合", "early_mlp": "早期融合 MLP"}
    miss_mod_name = {"text": "文本缺失", "image": "图像缺失", "both": "双模态缺失"}
    missing_ratio_table = md_table(
        ["模型", "缺失类型", "10%", "30%", "50%", "70%"],
        [[miss_model_name.get(m, m), miss_mod_name.get(mod, mod)] +
         [f"{get_miss(m, mod, r)['macro_f1_mean']:.4f}±{get_miss(m, mod, r)['macro_f1_std']:.4f}"
          for r in (0.1, 0.3, 0.5, 0.7)]
         for m in ("gated", "early_mlp") for mod in ("text", "image", "both")])
    pos_name = {"beginning": "开头", "middle": "中间", "end": "结尾"}
    missing_position_table = md_table(
        ["模态", "位置", "10%", "30%", "50%", "70%"],
        [[miss_mod_name.get(mod, mod), pos_name.get(pos, pos)] +
         [f"{next(r for r in missing_position['rows'] if r['model'] == 'gated' and r['modality'] == mod and r['position'] == pos and abs(float(r['ratio']) - ratio) < 1e-9)['macro_f1']:.4f}"
          for ratio in (0.1, 0.3, 0.5, 0.7)]
         for mod in ("text", "image") for pos in ("beginning", "middle", "end")])
    del_mod_name = {"text": "文本", "image": "图像"}
    deletion_table = md_table(
        ["模态", "k", "按重要性删除概率下降", "随机删除概率下降", "按重要性删除 Macro-F1", "随机删除 Macro-F1"],
        [[del_mod_name[mod], k,
          f"{get_del(mod, 'top', k)['mean_prob_drop']:.4f}",
          f"{get_del(mod, 'random', k)['mean_prob_drop']:.4f}",
          f"{get_del(mod, 'top', k)['macro_f1']:.4f}",
          f"{get_del(mod, 'random', k)['macro_f1']:.4f}"]
         for mod, ks in (("text", (1, 2, 3, 5)), ("image", (1, 2, 3)))
         for k in ks])

    vals.update({
        "topic_table": topic_table,
        "model_table": model_table,
        "robust_table": robust_table,
        "external_table": ext_table,
        "province_table": province_table,
        "hashtag_table": hashtag_table,
        "alert_table": alert_table,
        "gate_table": gate_table,
        "text_occ_table": text_occ_table,
        "missing_ratio_table": missing_ratio_table,
        "missing_position_table": missing_position_table,
        "deletion_table": deletion_table,
    })

    # ---------- 填充 ----------
    text = open(TEMPLATE, encoding="utf-8").read()
    for k, v in vals.items():
        text = text.replace("{{" + k + "}}", str(v))
    left = sorted(set(re.findall(r"\{\{([^}]+)\}\}", text)))
    if left:
        raise SystemExit("模板仍有未填充占位符：" + "、".join(left))
    os.makedirs(DOCS, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"报告 -> {os.path.relpath(REPORT, ROOT)}")
    print(f"  字数（去空白）约 {len(re.sub(r'\s', '', text)):,}")

    # ---------- 实验记录 ----------
    log = f"""# 实验记录（自动生成）

生成时间：{now_str()}  
数据 DOI：10.57760/sciencedb.j00133.00366（CC BY 4.0）  
固定随机种子：42

## 1. 数据与划分

| 项目 | 数值 |
|---|---:|
| 原始图文记录 | 64,347 |
| 清洗去重后全量 | {ana['data']['total_rows']:,} |
| 建模样本 | {ana['data']['sample_rows']:,} |
| 训练/验证/测试 | {met['data']['train']:,} / {met['data']['val']:,} / {met['data']['test']:,} |
| 外部无图测试 | {met['data']['external_no_image']:,} |
| 日期回填覆盖 | {pct(ana['data']['date_coverage'])} |
| 词表规模 | {ana['data']['vocab_size']:,} |

## 2. 特征与训练配置

- 文本：char_wb 2~4 gram TF-IDF，min_df=3，max_features=80,000，SVD 256 维，解释方差 {met['data']['text_svd_explained_variance']:.3f}。
- 图像：ResNet18 ImageNet 冻结 512 维（L2 归一化）+ 7 维统计（标准化×0.5），共 {met['data']['image_dim']} 维。
- 门控融合：双塔 + 样本级门控 + 缺失嵌入 + 模态 dropout 0.15 + 单模态 logit 残差。
- 训练：AdamW，lr=1.5e-3，weight decay=1e-4，batch=128，label smoothing=0.05，早停耐心 8。
- 对照：文本 LR、图像 LR、早期融合 LR、晚期融合（验证集选 alpha={met['config']['late_fusion_alpha']:.2f}）、早期融合 MLP、门控等权消融。

## 3. 完整模态结果

{model_table}

## 4. 缺失场景结果

{robust_table}

## 5. 外部无图结果

{ext_table}

## 6. 门控权重

{gate_table}

## 7. 文本遮挡 TOP8

{text_occ_table}

## 8. 图像 3×3 遮挡（仅图像场景，概率下降）

| 行\\列 | 左 | 中 | 右 |
|---|---:|---:|---:|
| 上 | {img_tl:.4f} | {img_grid[0][1]:.4f} | {img_grid[0][2]:.4f} |
| 中 | {img_grid[1][0]:.4f} | {img_center:.4f} | {img_grid[1][2]:.4f} |
| 下 | {img_grid[2][0]:.4f} | {img_grid[2][1]:.4f} | {img_br:.4f} |

## 8. 缺失比例与缺失位置（E2）

{missing_ratio_table}

缺失位置为特征维连续区间置零的代理实验（不是视频时间轴）：

{missing_position_table}

关键数字：门控文本缺失 10% Macro-F1 {f4(get_miss('gated', 'text', 0.1)['macro_f1_mean'])}，70% 降至 {f4(get_miss('gated', 'text', 0.7)['macro_f1_mean'])}；双模态缺失 10% 为 {f4(get_miss('gated', 'both', 0.1)['macro_f1_mean'])}，70% 降至 {f4(get_miss('gated', 'both', 0.7)['macro_f1_mean'])}；图像缺失 10%~70% 基本不变。位置代理中，文本开头缺失影响最大（70% 时 {f4(get_pos('text', 'beginning', 0.7)['macro_f1'])}），中间/结尾相对较小（{f4(get_pos('text', 'middle', 0.7)['macro_f1'])}/{f4(get_pos('text', 'end', 0.7)['macro_f1'])}）；图像三个位置差异很小。该结论只适用于静态特征代理，不能外推到视频时间轴。

## 9. 删除实验（E4）

{deletion_table}

文本删除 top-1 概率下降 {f4(get_del('text', 'top', 1)['mean_prob_drop'])}，随机删除 {f4(get_del('text', 'random', 1)['mean_prob_drop'])}；图像删除 top-1 为 {f4(get_del('image', 'top', 1)['mean_prob_drop'])}，随机删除 {f4(get_del('image', 'random', 1)['mean_prob_drop'])}。删除实验是敏感性分析，不是因果证明。

## 10. 单条推理与解释卡

- 文本单条推理耗时：{get_infer('text_only')} ms；
- 文本+图像单条推理耗时：{get_infer('text_image')} ms；
- 解释卡数量：{int(cards.get('n', 0))} 条（`output/explanation_cards.json`）；
- 离线页：`dashboard/single.html`；
- 强度为概率映射演示值，不是人工 −3~3 强度标注。

## 11. 运行与日志

| 步骤 | 日志/产物 | 说明 |
|---|---|---|
| 02 预处理 | `output/preprocess_report.json`、`output/preprocess_log.txt` | 图像特征 12,000×512 |
| 03 分析 | `output/analysis_log.txt`、`output/analysis_results.json` | 趋势/话题/关键词/预警 |
| 04 建模 | `output/modeling_log.txt`、`output/models_metrics.json` | 7 个模型 + 缺失场景 + 遮挡 |
| 12 缺失/删除实验 | `output/missing_experiments.json` | 缺失比例/位置 + 删除实验 |
| 10 单条推理 | `output/inference_self_test.json` | 文本/图文两条自检 |
| 11 解释卡 | `output/explanation_cards.json`、`dashboard/single.html` | 3 条示例 |
| 05 大屏 | `dashboard/data.js` | 14 个视图 |
| 06 图 | `docs/figures/fig1~fig12.png` | 报告插图 |
| 13 标注工具 | `output/annotation_self_test.json` | 合成数据自检 |
| 09 报告 | `docs/01_研究报告.md` | 本报告 |

## 12. 复现注意事项

1. 先运行 `python src/run_all.py`，再运行 `python src/07_export_docs.py --pdf`，最后 `python src/08_submission_check.py`。
2. 若重跑 03/04，必须重跑 12 → 10/11 → 05/06 → 09，否则大屏、图和报告数字会过期。
3. 日期覆盖率、外部测试集和 1,000 条落盘样本的口径见 `docs/03_数据说明.md`。
4. `run_all.py` 不使用 `--demo`；主模型不使用合成数据。
5. 所有指标基于固定 test split（1,800 条，三类各 600），bootstrap 1,000 次。
6. 标注 Kappa 自检使用合成数据，不是人工标注结果；项目尚未完成 1,500 条双人标注。
"""
    with open(EXP_LOG, "w", encoding="utf-8") as f:
        f.write(log)
    print(f"实验记录 -> {os.path.relpath(EXP_LOG, ROOT)}")


if __name__ == "__main__":
    main()
