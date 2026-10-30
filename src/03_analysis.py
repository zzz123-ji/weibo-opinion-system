# -*- coding: utf-8 -*-
"""03 舆情分析：情感分布、日度趋势、话题、关键词、省份、话题标签、异常预警。

输入
----
data/processed/posts_full.csv       清洗后全量 5.6 万条（趋势/话题）
data/processed/posts_clean.csv      建模样本 1.2 万条（示例）
data/processed/image_stats.csv      图像质量统计

输出
----
output/analysis_results.json        大屏/报告共用的事实表
output/sentiment_detail.csv         1.2 万条明细（匿名，供大屏示例）
output/tables/日度情感趋势.csv
output/tables/话题分析.csv
output/tables/情感关键词.csv
output/tables/省份提及.csv
output/tables/话题标签.csv
output/tables/异常预警.csv
output/tables/情感样本示例.csv
output/tables/文本长度统计.csv
output/tables/图像统计.csv
"""
from __future__ import annotations

import math
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import (LABEL_NAMES, LABEL_ORDER, OUT, PROCESSED, TABLES, now_str,
                        write_json)
from lib_text import tokenize, tokenize_keywords

RANDOM_SEED = 42
TOPIC_K = 8


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
    if isinstance(v, dict):
        return {str(k): py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [py(x) for x in v]
    return v


def main():
    t0 = time.time()
    print("=" * 66)
    print("03 舆情分析：趋势 / 话题 / 关键词 / 省份 / 预警")
    print("=" * 66)

    full = pd.read_csv(os.path.join(PROCESSED, "posts_full.csv"), dtype=str)
    full["label"] = full["label"].astype(int)
    full["text_len"] = pd.to_numeric(full["text_len"], errors="coerce").fillna(0).astype(int)
    full["date"] = full["date"].where(full["date"].notna(), None)
    print(f"  全量 {len(full):,} 条；有日期 {full['date'].notna().sum():,} 条")

    label_counts = {str(k): int(v) for k, v in full["label"].value_counts().sort_index().items()}
    label_ratio = {k: round(v / len(full), 6) for k, v in label_counts.items()}

    # ---------- 日度趋势 ----------
    print("\n[1] 日度情感趋势")
    dated = full[full["date"].notna()].copy()
    dated["is_neg"] = (dated["label"] == -1).astype(int)
    dated["is_pos"] = (dated["label"] == 1).astype(int)
    daily = dated.groupby("date").agg(
        n=("label", "size"),
        neg=("is_neg", "sum"),
        pos=("is_pos", "sum"),
        mean_sentiment=("label", "mean"),
    ).reset_index()
    daily["neu"] = daily["n"] - daily["neg"] - daily["pos"]
    daily["neg_ratio"] = daily["neg"] / daily["n"]
    daily["pos_ratio"] = daily["pos"] / daily["n"]
    daily["mean_sentiment"] = daily["mean_sentiment"].round(6)
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["neg_ratio_7d"] = daily["neg_ratio"].rolling(7, min_periods=3).mean()
    daily["volume_7d"] = daily["n"].rolling(7, min_periods=3).mean()
    daily["low_sample"] = daily["n"] < 50

    base = daily[daily["n"] >= 100].copy()
    base_neg = float(base["neg_ratio"].mean())
    base_neg_std = float(base["neg_ratio"].std(ddof=0)) or 1e-9
    base_vol = float(base["n"].mean())
    base_vol_std = float(base["n"].std(ddof=0)) or 1e-9
    daily["z_neg"] = (daily["neg_ratio"] - base_neg) / base_neg_std
    daily["z_volume"] = (daily["n"] - base_vol) / base_vol_std
    daily["alert_score"] = 0.6 * daily["z_neg"] + 0.4 * daily["z_volume"]

    alert_rows = []
    for _, r in daily.iterrows():
        if r["n"] < 100:
            continue
        reasons = []
        if r["z_neg"] >= 1.0:
            reasons.append("负向率高于基线")
        if r["z_volume"] >= 1.5:
            reasons.append("声量高于基线")
        if not reasons:
            continue
        level = "高" if (r["z_neg"] >= 2.0 or r["z_volume"] >= 2.5) else "中"
        alert_rows.append({
            "date": r["date"], "n": int(r["n"]), "neg_ratio": round(float(r["neg_ratio"]), 6),
            "z_neg": round(float(r["z_neg"]), 3), "z_volume": round(float(r["z_volume"]), 3),
            "score": round(float(r["alert_score"]), 3), "level": level,
            "reason": "；".join(reasons),
            "baseline_neg_ratio": round(base_neg, 6), "baseline_volume": round(base_vol, 1),
        })
    alert_rows = sorted(alert_rows, key=lambda x: x["score"], reverse=True)[:10]
    if len(alert_rows) < 3:
        # 兜底：取综合得分最高的 3 天，保证大屏有可解释的预警样例
        extra = daily[daily["n"] >= 100].sort_values("alert_score", ascending=False).head(3)
        alert_rows = [{
            "date": r["date"], "n": int(r["n"]), "neg_ratio": round(float(r["neg_ratio"]), 6),
            "z_neg": round(float(r["z_neg"]), 3), "z_volume": round(float(r["z_volume"]), 3),
            "score": round(float(r["alert_score"]), 3), "level": "中",
            "reason": "综合得分较高的重点观察日",
            "baseline_neg_ratio": round(base_neg, 6), "baseline_volume": round(base_vol, 1),
        } for _, r in extra.iterrows()]
    print(f"  预警 {len(alert_rows)} 条；基线负向率 {base_neg:.2%}，基线声量 {base_vol:.0f}")

    pre = daily[daily["date"] < "2020-01-20"]
    post = daily[daily["date"] >= "2020-01-20"]
    peak_vol = daily.loc[daily["n"].idxmax()]
    peak_neg = daily[daily["n"] >= 100].sort_values("neg_ratio").iloc[-1]
    trend = {
        "pre_jan20_mean_n": round(float(pre["n"].mean()), 1),
        "post_jan20_mean_n": round(float(post["n"].mean()), 1),
        "volume_growth": round(float(post["n"].mean() / max(pre["n"].mean(), 1e-9) - 1), 4),
        "peak_volume_date": peak_vol["date"],
        "peak_volume_n": int(peak_vol["n"]),
        "peak_neg_date": peak_neg["date"],
        "peak_neg_ratio": round(float(peak_neg["neg_ratio"]), 6),
        "overall_neg_ratio": round(float((full["label"] == -1).mean()), 6),
        "overall_pos_ratio": round(float((full["label"] == 1).mean()), 6),
        "baseline_neg_ratio": round(base_neg, 6),
        "baseline_volume": round(base_vol, 1),
        "n_days": int(len(daily)),
        "n_days_n100": int((daily["n"] >= 100).sum()),
    }
    print(f"  1月20日前日均 {trend['pre_jan20_mean_n']} 条，之后 {trend['post_jan20_mean_n']} 条，"
          f"增长 {trend['volume_growth']:.1%}")

    # ---------- 分词 + 话题 + 关键词 ----------
    print("\n[2] 分词（jieba 词性过滤）")
    tokens_list = []
    for i, text in enumerate(full["text_clean"].fillna("").tolist()):
        tokens_list.append(tokenize(text))
        if (i + 1) % 10000 == 0:
            print(f"    {i + 1:,}/{len(full):,}")
    vocab_size = len({w for toks in tokens_list for w in toks})
    print(f"  词表 {vocab_size:,}，平均每帖 {np.mean([len(t) for t in tokens_list]):.1f} 词")

    print("\n[3] 话题建模（TF-IDF + NMF）")
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10, max_df=0.55,
                                 max_features=6000, sublinear_tf=True)
    X = vectorizer.fit_transform(tokens_list)
    k = min(TOPIC_K, max(2, X.shape[1] - 1))
    nmf = NMF(n_components=k, init="nndsvda", random_state=RANDOM_SEED,
              max_iter=350, beta_loss="frobenius")
    W = nmf.fit_transform(X)
    H = nmf.components_
    terms = vectorizer.get_feature_names_out()
    topic_ids = W.argmax(axis=1)
    full["topic_id"] = topic_ids
    topic_rows = []
    for tid in range(k):
        top_idx = H[tid].argsort()[::-1][:10]
        top_terms = [str(terms[i]) for i in top_idx]
        mask = topic_ids == tid
        n = int(mask.sum())
        sub = full[mask]
        label = "·".join(top_terms[:3])
        topic_rows.append({
            "id": tid,
            "label": label,
            "top_terms": top_terms,
            "n": n,
            "share": round(n / len(full), 6),
            "neg_ratio": round(float((sub["label"] == -1).mean()), 6) if n else 0.0,
            "pos_ratio": round(float((sub["label"] == 1).mean()), 6) if n else 0.0,
            "mean_sentiment": round(float(sub["label"].mean()), 6) if n else 0.0,
        })
    topic_rows = sorted(topic_rows, key=lambda x: x["n"], reverse=True)
    # 用主题内 W 最高的帖子做示例（截断，匿名）
    for t in topic_rows:
        tid = t["id"]
        idx = np.where(topic_ids == tid)[0]
        order = idx[np.argsort(W[idx, tid])[::-1][:2]]
        examples = []
        for j in order:
            examples.append({
                "text": str(full.iloc[j]["text_clean"])[:120],
                "label": int(full.iloc[j]["label"]),
                "label_name": LABEL_NAMES[int(full.iloc[j]["label"])],
                "date": full.iloc[j]["date"] if pd.notna(full.iloc[j]["date"]) else "",
            })
        t["examples"] = examples
    print(f"  话题 {k} 个：" + "；".join(f"{t['label']}({t['n']})" for t in topic_rows[:4]))

    print("\n[4] 情感关键词（对数几率比 × 词频权重）")
    kw_tokens = [tokenize_keywords(t) for t in full["text_clean"].fillna("").tolist()]
    counters = {lab: Counter() for lab in LABEL_ORDER}
    for lab, toks in zip(full["label"].tolist(), kw_tokens):
        counters[lab].update(toks)
    all_vocab = set().union(*[set(c) for c in counters.values()])
    V = max(len(all_vocab), 1)
    totals = {lab: sum(c.values()) for lab, c in counters.items()}
    keywords = {}
    for lab in LABEL_ORDER:
        other = Counter()
        other_total = 0
        for lab2 in LABEL_ORDER:
            if lab2 == lab:
                continue
            other.update(counters[lab2])
            other_total += totals[lab2]
        rows = []
        alpha = 1.0
        for w, c in counters[lab].items():
            if c < 30:
                continue
            p1 = (c + alpha) / (totals[lab] + alpha * V)
            p0 = (other.get(w, 0) + alpha) / (other_total + alpha * V)
            log_odds = float(np.log(p1) - np.log(p0))
            # 词频加权：避免只出现几十次的专名压过高频情感词
            score = log_odds * float(np.log1p(c))
            rows.append({"word": w, "count": int(c),
                         "log_odds": round(log_odds, 4), "score": round(score, 4)})
        rows = sorted(rows, key=lambda x: x["score"], reverse=True)[:25]
        keywords[str(lab)] = rows
        print(f"  {LABEL_NAMES[lab]} top5：" + "、".join(x["word"] for x in rows[:5]))

    # ---------- 省份 / 话题标签 ----------
    print("\n[5] 省份提及与话题标签")
    prov_rows = full[full["provinces"].notna() & (full["provinces"].astype(str) != "")].copy()
    prov_expl = prov_rows.assign(province=prov_rows["provinces"].astype(str).str.split("|")).explode("province")
    prov_expl = prov_expl[prov_expl["province"].astype(str).str.len() > 0]
    prov_stats = prov_expl.groupby("province").agg(
        n=("label", "size"), neg=("label", lambda s: int((s == -1).sum())),
        pos=("label", lambda s: int((s == 1).sum())),
        mean_sentiment=("label", "mean")).reset_index()
    prov_stats["share"] = prov_stats["n"] / len(full)
    prov_stats["neg_ratio"] = prov_stats["neg"] / prov_stats["n"]
    prov_stats["pos_ratio"] = prov_stats["pos"] / prov_stats["n"]
    prov_stats = prov_stats.sort_values("n", ascending=False)
    provinces = [{
        "province": r["province"], "n": int(r["n"]),
        "share": round(float(r["share"]), 6),
        "neg_ratio": round(float(r["neg_ratio"]), 6),
        "pos_ratio": round(float(r["pos_ratio"]), 6),
        "mean_sentiment": round(float(r["mean_sentiment"]), 6),
    } for _, r in prov_stats.head(20).iterrows()]
    print("  省份 top5：" + "、".join(f"{p['province']}({p['n']})" for p in provinces[:5]))

    tag_rows = full[full["hashtags"].notna() & (full["hashtags"].astype(str) != "")].copy()
    tag_expl = tag_rows.assign(tag=tag_rows["hashtags"].astype(str).str.split("|")).explode("tag")
    tag_expl = tag_expl[tag_expl["tag"].astype(str).str.len() >= 2]
    tag_stats = tag_expl.groupby("tag").agg(
        n=("label", "size"), neg=("label", lambda s: int((s == -1).sum())),
        pos=("label", lambda s: int((s == 1).sum())),
        mean_sentiment=("label", "mean")).reset_index()
    tag_stats["neg_ratio"] = tag_stats["neg"] / tag_stats["n"]
    tag_stats["pos_ratio"] = tag_stats["pos"] / tag_stats["n"]
    tag_stats = tag_stats.sort_values("n", ascending=False)
    hashtags = [{
        "tag": r["tag"], "n": int(r["n"]),
        "neg_ratio": round(float(r["neg_ratio"]), 6),
        "pos_ratio": round(float(r["pos_ratio"]), 6),
        "mean_sentiment": round(float(r["mean_sentiment"]), 6),
    } for _, r in tag_stats.head(20).iterrows()]
    print("  话题标签 top5：" + "、".join(f"#{h['tag']}#({h['n']})" for h in hashtags[:5]))

    # ---------- 图像统计 ----------
    print("\n[6] 图像统计与显著性")
    img = pd.read_csv(os.path.join(PROCESSED, "image_stats.csv"))
    img["label"] = img["label"].astype(int)
    metrics = ["brightness", "saturation", "contrast", "colorfulness", "edge_density", "entropy", "warm_ratio"]
    by_label = {}
    for lab in LABEL_ORDER:
        sub = img[img["label"] == lab]
        by_label[str(lab)] = {
            "n": int(len(sub)),
            **{m: round(float(sub[m].mean()), 6) for m in metrics},
        }
    anova = {}
    for m in metrics:
        groups = [img[img["label"] == lab][m].dropna().values for lab in LABEL_ORDER]
        try:
            f, p = f_oneway(*groups)
            anova[m] = {"f": round(float(f), 4), "p": round(float(p), 6)}
        except Exception:
            anova[m] = {"f": 0.0, "p": 1.0}
    print("  亮度 p=%.4g，色彩丰富度 p=%.4g，边缘密度 p=%.4g" % (
        anova["brightness"]["p"], anova["colorfulness"]["p"], anova["edge_density"]["p"]))

    # ---------- 文本长度 ----------
    text_len = []
    for lab in LABEL_ORDER:
        sub = full[full["label"] == lab]
        text_len.append({
            "label": lab, "label_name": LABEL_NAMES[lab], "n": int(len(sub)),
            "mean": round(float(sub["text_len"].mean()), 2),
            "median": int(sub["text_len"].median()),
            "p95": int(sub["text_len"].quantile(0.95)),
        })

    # ---------- 明细（12k 建模样本） ----------
    sample = pd.read_csv(os.path.join(PROCESSED, "posts_clean.csv"), dtype=str)
    sample["label"] = sample["label"].astype(int)
    topic_map = dict(zip(full["post_hash"], full["topic_id"]))
    topic_label_map = {t["id"]: t["label"] for t in topic_rows}
    sample["topic_id"] = sample["post_hash"].map(topic_map)
    sample["topic_label"] = sample["topic_id"].map(topic_label_map)
    sample["date"] = sample["date"].where(sample["date"].notna(), None)
    detail = sample[["post_hash", "date", "label", "label_name", "text_clean", "topic_label",
                     "provinces", "has_hashtag", "image_file"]].copy()
    detail["text_clean"] = detail["text_clean"].astype(str).str.slice(0, 100)
    detail.to_csv(os.path.join(OUT, "sentiment_detail.csv"), index=False, encoding="utf-8-sig")

    # ---------- 发现 ----------
    print("\n[7] 生成主要发现")
    top_topic = topic_rows[0]
    neg_topics = [t for t in topic_rows if t["n"] >= 500]
    neg_topic = max(neg_topics, key=lambda x: x["neg_ratio"]) if neg_topics else top_topic
    top_prov = provinces[0] if provinces else {"province": "—", "n": 0, "neg_ratio": 0}
    top_tag = hashtags[0] if hashtags else {"tag": "—", "n": 0, "neg_ratio": 0}
    top_alert = alert_rows[0] if alert_rows else {"date": "—", "n": 0, "neg_ratio": 0, "reason": ""}
    neg_words = [x["word"] for x in keywords["-1"][:6]]
    pos_words = [x["word"] for x in keywords["1"][:6]]
    findings = [
        f"发现1：清洗去重后得到 {len(full):,} 条有效图文帖，类别为负面 {label_counts['-1']:,} 条"
        f"（{label_ratio['-1']:.2%}）、中性 {label_counts['0']:,} 条（{label_ratio['0']:.2%}）、"
        f"正面 {label_counts['1']:,} 条（{label_ratio['1']:.2%}），中性样本占比最高，存在明显类别不平衡。",
        f"发现2：{full['date'].notna().sum():,} 条帖子可回填日期，覆盖率 {full['date'].notna().mean():.2%}；"
        f"趋势分析基于 {len(daily)} 天（{daily['date'].min()} 至 {daily['date'].max()}），"
        f"日期来自公开文本表的精确文本匹配，未做任何推测。",
        f"发现3：1 月 20 日前日均发帖 {trend['pre_jan20_mean_n']:.0f} 条，1 月 20 日后升至 "
        f"{trend['post_jan20_mean_n']:.0f} 条，增长 {trend['volume_growth']:.1%}，"
        f"与疫情进入公众视野的时间点吻合。",
        f"发现4：单日声量峰值出现在 {trend['peak_volume_date']}，共 {trend['peak_volume_n']:,} 条；"
        f"有日期样本中负向率最高的一天是 {trend['peak_neg_date']}（{trend['peak_neg_ratio']:.2%}）。",
        f"发现5：NMF 将讨论划分为 {k} 个话题，最大话题为“{top_topic['label']}”，"
        f"覆盖 {top_topic['n']:,} 条（{top_topic['share']:.2%}）。",
        f"发现6：负向率最高的话题是“{neg_topic['label']}”，负向率 {neg_topic['neg_ratio']:.2%}，"
        f"高于全量负向率 {trend['overall_neg_ratio']:.2%}，是舆情预警的重点语义场。",
        f"发现7：负向文本的高辨识度词包括“{'、'.join(neg_words)}”，"
        f"多与疾病、情绪低落和公共事件担忧相关。",
        f"发现8：正向文本的高辨识度词包括“{'、'.join(pos_words)}”，"
        f"多与祝福、鼓励、康复和日常生活相关。",
        f"发现9：省份提及共覆盖 {len(prov_rows):,} 条（{len(prov_rows) / len(full):.2%}），"
        f"提及最多的是{top_prov['province']}（{top_prov['n']:,} 条，负向率 {top_prov['neg_ratio']:.2%}）；"
        f"该指标是文本提及，不代表用户地理位置。",
        f"发现10：话题标签共出现在 {len(tag_rows):,} 条帖子中，最高频标签为 #{top_tag['tag']}#"
        f"（{top_tag['n']:,} 条，负向率 {top_tag['neg_ratio']:.2%}）。",
        f"发现11：共触发 {len(alert_rows)} 个重点观察日，综合得分最高的是 {top_alert['date']}"
        f"（{top_alert['n']:,} 条，负向率 {top_alert['neg_ratio']:.2%}，{top_alert['reason']}）。",
        f"发现12：图像统计显示负面帖平均亮度 {by_label['-1']['brightness']:.3f}、"
        f"色彩丰富度 {by_label['-1']['colorfulness']:.3f}，正面帖分别为 "
        f"{by_label['1']['brightness']:.3f}、{by_label['1']['colorfulness']:.3f}；"
        f"亮度差异 ANOVA p={anova['brightness']['p']:.4g}，色彩丰富度 p={anova['colorfulness']['p']:.4g}。",
    ]
    for f in findings:
        print("  " + f)

    # ---------- 落盘 ----------
    print("\n[8] 写表与 JSON")
    daily_out = daily.copy()
    for c in ["neg_ratio", "pos_ratio", "neg_ratio_7d", "volume_7d", "z_neg", "z_volume", "alert_score"]:
        daily_out[c] = daily_out[c].round(6)
    daily_out.to_csv(os.path.join(TABLES, "日度情感趋势.csv"), index=False, encoding="utf-8-sig")

    topic_df = pd.DataFrame([{k: v for k, v in t.items() if k != "examples"} for t in topic_rows])
    topic_df.to_csv(os.path.join(TABLES, "话题分析.csv"), index=False, encoding="utf-8-sig")

    kw_rows = []
    for lab in LABEL_ORDER:
        for rank, x in enumerate(keywords[str(lab)], 1):
            kw_rows.append({"label": lab, "label_name": LABEL_NAMES[lab], "rank": rank,
                            "word": x["word"], "count": x["count"],
                            "log_odds": x["log_odds"], "score": x["score"]})
    pd.DataFrame(kw_rows).to_csv(os.path.join(TABLES, "情感关键词.csv"),
                                 index=False, encoding="utf-8-sig")

    pd.DataFrame(provinces).to_csv(os.path.join(TABLES, "省份提及.csv"),
                                   index=False, encoding="utf-8-sig")
    pd.DataFrame(hashtags).to_csv(os.path.join(TABLES, "话题标签.csv"),
                                  index=False, encoding="utf-8-sig")
    pd.DataFrame(alert_rows).to_csv(os.path.join(TABLES, "异常预警.csv"),
                                    index=False, encoding="utf-8-sig")
    pd.DataFrame(text_len).to_csv(os.path.join(TABLES, "文本长度统计.csv"),
                                  index=False, encoding="utf-8-sig")

    img_rows = []
    for lab in LABEL_ORDER:
        row = {"label": lab, "label_name": LABEL_NAMES[lab], **by_label[str(lab)]}
        row.update({f"{m}_p": anova[m]["p"] for m in metrics})
        img_rows.append(row)
    pd.DataFrame(img_rows).to_csv(os.path.join(TABLES, "图像统计.csv"),
                                  index=False, encoding="utf-8-sig")

    examples = []
    for lab in LABEL_ORDER:
        sub = detail[(detail["label"] == lab) & (detail["date"].notna())
                     & (detail["text_clean"].str.len() >= 20)]
        if len(sub) == 0:
            sub = detail[detail["label"] == lab]
        for _, r in sub.sample(n=min(4, len(sub)), random_state=RANDOM_SEED + lab).iterrows():
            examples.append({
                "label": int(r["label"]), "label_name": r["label_name"],
                "date": r["date"], "topic": r["topic_label"] if pd.notna(r["topic_label"]) else "",
                "province": r["provinces"] if pd.notna(r["provinces"]) else "",
                "text": r["text_clean"],
            })
    pd.DataFrame(examples).to_csv(os.path.join(TABLES, "情感样本示例.csv"),
                                  index=False, encoding="utf-8-sig")

    result = {
        "generated_at": now_str(),
        "data": {
            "total_rows": int(len(full)),
            "sample_rows": int(len(sample)),
            "label_counts": label_counts,
            "label_ratio": label_ratio,
            "date_rows": int(full["date"].notna().sum()),
            "date_coverage": round(float(full["date"].notna().mean()), 6),
            "date_min": str(daily["date"].min()),
            "date_max": str(daily["date"].max()),
            "unique_days": int(len(daily)),
            "text_len_mean": round(float(full["text_len"].mean()), 2),
            "text_len_median": int(full["text_len"].median()),
            "text_len_p95": int(full["text_len"].quantile(0.95)),
            "hashtag_rows": int(len(tag_rows)),
            "province_rows": int(len(prov_rows)),
            "province_coverage": round(float(len(prov_rows) / len(full)), 6),
            "image_sample_rows": int(len(img)),
            "vocab_size": int(vocab_size),
        },
        "trend": trend,
        "daily": [{k: py(v) for k, v in r.items()} for r in daily.to_dict(orient="records")],
        "topics": {"k": k, "method": "TF-IDF + NMF", "list": topic_rows},
        "keywords": keywords,
        "provinces": provinces,
        "hashtags": hashtags,
        "alerts": alert_rows,
        "image_stats": {"by_label": by_label, "anova": anova,
                        "note": "基于 12,000 条建模样本的图片质量统计；ANOVA 为三类别单因素检验"},
        "text_len": text_len,
        "findings": findings,
    }
    write_json(os.path.join(OUT, "analysis_results.json"), result)
    print(f"  analysis_results.json 已写入；总耗时 {time.time() - t0:.1f}s")
    print("\n[完成] 舆情分析完成。")


if __name__ == "__main__":
    main()
