# -*- coding: utf-8 -*-
"""11 导出单条分析解释卡：output/explanation_cards.json + dashboard/single.html。

用途
----
- 把 10_inference_service 的推理结果固化成可离线打开的示例页；
- 三条示例分别覆盖负面/中性/正面，图片来自 data/crawl/images；
- 解释卡包含：预测、概率、门控权重、强度演示值、文本 top token、图像 3×3 遮挡。

边界
----
- 强度是三类概率映射的演示值，不是人工 -3~3 强度标注；
- 示例只用于展示单条推理链路，不代表总体准确率。
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import LABEL_NAMES, OUT, ROOT
from lib_infer import SentimentInference

CRAWL = os.path.join(ROOT, "data", "crawl")
DASH = os.path.join(ROOT, "dashboard")
OUT_JSON = os.path.join(OUT, "explanation_cards.json")
OUT_HTML = os.path.join(DASH, "single.html")


def pick_examples(engine: SentimentInference, max_candidates: int = 300):
    """每个真实类别优先选“预测正确且置信度最高”的样本。

    说明：这是展示页的挑选策略，只用于让三条示例覆盖三类且便于核对解释链路，
    不代表总体准确率；完整指标仍以 output/models_metrics.json 为准。
    """
    candidates = []
    with open(os.path.join(CRAWL, "weibo_1000.csv"), encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            img = row.get("image_file", "")
            if not img or not os.path.exists(os.path.join(CRAWL, img.replace("/", os.sep))):
                continue
            candidates.append(row)
            if len(candidates) >= max_candidates:
                break
    best = {}
    fallback = {}
    for row in candidates:
        image_path = os.path.join("data", "crawl", row["image_file"])
        res = engine.predict(row["text"], image_path=image_path, explain=False)
        pred = res["prediction"]
        conf = max(pred["probabilities"].values())
        lab = int(row["label"])
        item = (conf, row, pred)
        if lab not in fallback or conf > fallback[lab][0]:
            fallback[lab] = item
        if pred["label"] == lab and (lab not in best or conf > best[lab][0]):
            best[lab] = item
    picked = []
    for lab in (-1, 0, 1):
        item = best.get(lab) or fallback.get(lab)
        if item:
            picked.append(item[1])
    return picked


def build_cards(engine: SentimentInference, rows: list[dict]) -> list[dict]:
    cards = []
    for row in rows:
        image_path = os.path.join("data", "crawl", row["image_file"])
        res = engine.predict(row["text"], image_path=image_path, explain=True)
        cards.append({
            "post_id": row["post_id"],
            "true_label": int(row["label"]),
            "true_label_name": LABEL_NAMES[int(row["label"])],
            "text": row["text"],
            "image_file": row["image_file"],
            "image_url": "../data/crawl/" + row["image_file"].replace("\\", "/"),
            "prediction": res["prediction"],
            "modalities": res["modalities"],
            "evidence": res.get("evidence", {}),
            "latency_ms": res.get("latency_ms"),
            "caveats": res["caveats"],
        })
    return cards


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>单条分析 · 微博多模态情感解释卡</title>
<style>
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:#050b18; color:#cfe3ff; font-family:"Microsoft YaHei","PingFang SC",sans-serif;
    min-height:100vh; padding:26px 34px 40px; }
  header { max-width:1500px; margin:0 auto 18px; border-bottom:1px solid #17396b; padding-bottom:14px; }
  h1 { font-size:26px; color:#eaf4ff; letter-spacing:2px; }
  .sub { color:#5f8fce; font-size:12px; margin-top:6px; }
  .sub b { color:#00e4ff; font-weight:400; }
  .note { max-width:1500px; margin:0 auto 14px; font-size:12px; color:#7ea2d8;
    background:rgba(13,32,66,.65); border:1px solid #17396b; border-radius:6px; padding:9px 12px; }
  main { max-width:1500px; margin:0 auto; display:grid; gap:14px; }
  .card { display:grid; grid-template-columns:300px 1fr; gap:18px; background:linear-gradient(180deg,rgba(11,26,54,.95),rgba(6,15,33,.95));
    border:1px solid #163462; border-radius:8px; padding:16px; }
  .media img { width:100%; border-radius:6px; border:1px solid #244a7d; background:#0a1730; }
  .media .ph { height:220px; display:flex; align-items:center; justify-content:center;
    border:1px dashed #244a7d; border-radius:6px; color:#4d6f9e; font-size:12px; }
  .meta { font-size:11px; color:#4d6f9e; margin-top:8px; line-height:1.7; }
  .meta code { color:#9fc3ef; }
  .head { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }
  .title { font-size:15px; color:#9fd0ff; }
  .pred { font-size:20px; font-weight:700; }
  .pred.neg { color:#ff5d73; } .pred.neu { color:#8fa3c8; } .pred.pos { color:#2ee6a8; }
  .badge { display:inline-block; font-size:11px; padding:2px 8px; border-radius:10px;
    border:1px solid #2b4f83; color:#9fc3ef; margin-left:8px; vertical-align:middle; }
  .badge.ok { border-color:#1d7f63; color:#2ee6a8; }
  .badge.bad { border-color:#8e3144; color:#ff8fa2; }
  .text { font-size:13px; line-height:1.8; color:#d7e7ff; margin:10px 0 12px;
    max-height:92px; overflow:auto; background:#071226; border:1px solid #122c57; border-radius:6px; padding:9px 11px; }
  .cols { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  .box { background:#071226; border:1px solid #122c57; border-radius:6px; padding:9px 11px; }
  .box h4 { font-size:12px; color:#9fd0ff; font-weight:600; margin-bottom:8px; }
  .bar { height:16px; background:#0d1d38; border-radius:3px; overflow:hidden; margin:4px 0 7px; }
  .bar > i { display:block; height:100%; }
  .barrow { font-size:11px; color:#8fb2df; display:flex; justify-content:space-between; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; }
  .chip { font-size:11px; color:#cfe3ff; background:#102a50; border:1px solid #244a7d;
    border-radius:12px; padding:2px 9px; }
  .chip b { color:#ffb347; font-weight:600; }
  .grid9 { display:grid; grid-template-columns:repeat(3,1fr); gap:3px; width:150px; }
  .grid9 div { aspect-ratio:1; border-radius:2px; background:#0d1d38; border:1px solid #1b3b67;
    display:flex; align-items:center; justify-content:center; font-size:9px; color:#8fb2df; }
  .kv { font-size:11px; color:#8fb2df; line-height:1.9; }
  .kv b { color:#eaf4ff; font-weight:500; }
  .caveat { font-size:10.5px; color:#4d6f9e; margin-top:10px; line-height:1.7; }
  footer { max-width:1500px; margin:20px auto 0; font-size:10.5px; color:#3d5f8e; border-top:1px solid #122c57; padding-top:10px; }
  a { color:#00e4ff; text-decoration:none; }
</style>
</head>
<body>
<header>
  <h1>单条分析 · 微博多模态情感解释卡</h1>
  <div class="sub">文本 + 图像 · 门控融合 · 缺失场景可用 · <b>离线示例</b> · <a href="index.html">返回舆情大屏</a></div>
</header>
<div class="note">
  本页展示 3 条落盘样本的单条推理结果，用于说明“预测 + 门控权重 + 文本 token 证据 + 图像 3×3 遮挡”的解释链路。
  示例按“每个真实类别优先选预测正确且置信度高”的展示策略挑选；强度是三类概率映射的演示值，
  不是人工 -3~3 强度标注；示例不代表总体准确率，完整指标见 output/models_metrics.json。
</div>
<main id="cards"></main>
<footer>数据来源：ScienceDB《Weibo and Twitter datasets》DOI 10.57760/sciencedb.j00133.00366，CC BY 4.0。仅聚合舆情辅助，不用于个体判断或自动处置。</footer>
<script>
const DATA = __DATA__;
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cls = n => n === '负面' ? 'neg' : (n === '正面' ? 'pos' : 'neu');
const color = n => n === '负面' ? '#ff5d73' : (n === '正面' ? '#2ee6a8' : '#8fa3c8');
function probBars(p) {
  return Object.entries(p).map(([k,v]) =>
    `<div class="barrow"><span>${esc(k)}</span><span>${(v*100).toFixed(1)}%</span></div>
     <div class="bar"><i style="width:${(v*100).toFixed(1)}%;background:${color(k)}"></i></div>`).join('');
}
function gateBars(g) {
  return `<div class="barrow"><span>文本</span><span>${(g.text*100).toFixed(1)}%</span></div>
    <div class="bar"><i style="width:${(g.text*100).toFixed(1)}%;background:#00e4ff"></i></div>
    <div class="barrow"><span>图像</span><span>${(g.image*100).toFixed(1)}%</span></div>
    <div class="bar"><i style="width:${(g.image*100).toFixed(1)}%;background:#b78cff"></i></div>`;
}
function tokens(ev) {
  const arr = ev && ev.text_top_tokens ? ev.text_top_tokens : [];
  if (!arr.length) return '<span class="kv">无可用 token 证据</span>';
  return '<div class="chips">' + arr.map(t =>
    `<span class="chip">${esc(t.token)} <b>${t.prob_drop >= 0 ? '+' : ''}${Number(t.prob_drop).toFixed(4)}</b></span>`).join('') + '</div>';
}
function grid9(ev) {
  const arr = ev && ev.image_grid ? ev.image_grid : null;
  if (!arr) return '<span class="kv">无图像或未计算</span>';
  const mx = Math.max(0.0001, ...arr.map(x => Math.abs(x.prob_drop)));
  return '<div class="grid9">' + arr.map(x => {
    const a = Math.min(1, Math.abs(x.prob_drop) / mx);
    const bg = x.prob_drop >= 0 ? `rgba(255,93,115,${0.12 + 0.78*a})` : `rgba(0,228,255,${0.10 + 0.55*a})`;
    return `<div style="background:${bg}" title="行${x.row+1}列${x.col+1}：${x.prob_drop}">${x.prob_drop.toFixed(3)}</div>`;
  }).join('') + '</div>';
}
document.getElementById('cards').innerHTML = DATA.map((c, i) => {
  const p = c.prediction;
  const ok = p.label === c.true_label;
  return `<section class="card">
    <div class="media">
      ${c.image_url ? `<img src="${esc(c.image_url)}" alt="配图">` : '<div class="ph">无配图</div>'}
      <div class="meta">样本 ${i+1} · <code>${esc(c.post_id)}</code><br>
        真实标签：<b>${esc(c.true_label_name)}</b> · 图片：<code>${esc(c.image_file)}</code></div>
    </div>
    <div>
      <div class="head">
        <div>
          <div class="title">预测结果 <span class="badge ${ok?'ok':'bad'}">${ok?'与真实标签一致':'与真实标签不一致'}</span></div>
          <div class="pred ${cls(p.label_name)}">${esc(p.label_name)} <span class="badge">强度演示 ${p.intensity}</span></div>
        </div>
        <div class="kv">单条耗时 <b>${c.latency_ms ?? '—'} ms</b><br>主参考模态 <b>${esc(c.modalities.main_modality)}</b></div>
      </div>
      <div class="text">${esc(c.text)}</div>
      <div class="cols">
        <div class="box"><h4>三类概率</h4>${probBars(p.probabilities)}</div>
        <div class="box"><h4>门控权重</h4>${gateBars(c.modalities.gate)}
          <div class="kv" style="margin-top:6px">缺失时门控自动切换到可用模态；双缺失退化为先验。</div></div>
      </div>
      <div class="cols" style="margin-top:12px">
        <div class="box"><h4>文本证据 · 删除后概率下降 TOP3</h4>${tokens(c.evidence)}</div>
        <div class="box"><h4>图像证据 · 3×3 遮挡（仅图像场景）</h4>${grid9(c.evidence)}</div>
      </div>
      <div class="caveat">${c.caveats.map(esc).join(' ')}</div>
    </div>
  </section>`;
}).join('');
</script>
</body>
</html>
"""


def main():
    engine = SentimentInference()
    rows = pick_examples(engine)
    if not rows:
        raise SystemExit("data/crawl/weibo_1000.csv 中没有可用图片样本")
    cards = build_cards(engine, rows)
    payload = {
        "generated_at": engine.loaded_at,
        "n": len(cards),
        "method": "单条门控推理 + token 删除遮挡 + 图像 3×3 遮挡",
        "selection": "每个真实类别优先选预测正确且置信度最高的样本；仅用于展示解释链路，不代表总体准确率。",
        "caveat": "强度为概率映射演示值，不是人工 -3~3 强度标注；示例不代表总体准确率。",
        "cards": cards,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    data_js = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    html = HTML.replace("__DATA__", data_js)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] {len(cards)} 张解释卡 -> {os.path.relpath(OUT_JSON, ROOT)}")
    print(f"     离线单条页 -> {os.path.relpath(OUT_HTML, ROOT)}")
    for c in cards:
        print(f"  {c['post_id']}: true={c['true_label_name']} pred={c['prediction']['label_name']} "
              f"gate={c['modalities']['gate']} latency={c['latency_ms']}ms")


if __name__ == "__main__":
    main()
