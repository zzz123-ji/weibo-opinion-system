# -*- coding: utf-8 -*-
"""单条推理核心：加载门控融合模型，输出极性、概率、门控权重与解释卡。

设计边界
--------
- 只支持本项目训练时的两种模态：文本 + 图像；不伪装支持视频/语音；
- 强度字段是“三类概率映射”的演示值，不是人工 -3~3 强度标注；
- 解释采用 token 删除遮挡与图像 3×3 区域遮挡，和训练/评估口径一致；
- 模型、特征器、标签口径全部从 output/models 与 data/processed 读取，不重复训练。
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import LABEL_NAMES, MODELS, ROOT
from lib_image import ImageEncoder, image_stats, load_image
from lib_text import clean_text, tokenize_keywords

SEED = 42
CLASSES = np.array([-1, 0, 1])
STAT_COLS = ["brightness", "saturation", "contrast", "colorfulness",
             "edge_density", "entropy", "warm_ratio"]


class GatedFusionNet(nn.Module):
    """与 04_modeling.py 保持一致的网络结构（只用于加载权重）。"""

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
        return logits, g


def _read_image(image_path: str | None, image_base64: str | None) -> Image.Image | None:
    if image_base64:
        raw = image_base64.split(",", 1)[-1]
        try:
            return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
        except Exception:
            return None
    if image_path:
        path = image_path if os.path.isabs(image_path) else os.path.join(ROOT, image_path)
        return load_image(path)
    return None


class SentimentInference:
    """加载一次，重复预测；适合 FastAPI 进程内复用。"""

    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or MODELS
        baseline_path = os.path.join(self.model_dir, "baselines.joblib")
        gated_path = os.path.join(self.model_dir, "gated_fusion.pt")
        if not os.path.exists(baseline_path) or not os.path.exists(gated_path):
            raise FileNotFoundError("缺少 output/models/baselines.joblib 或 gated_fusion.pt，请先运行 04_modeling.py")
        pack = joblib.load(baseline_path)
        self.vectorizer = pack["vectorizer"]
        self.svd = pack["svd"]
        self.text_scaler = pack["text_scaler"]
        self.stats_scaler = pack["stats_scaler"]
        self.stat_cols = pack.get("stat_cols", STAT_COLS)
        self.classes = np.array(pack.get("classes", [-1, 0, 1]))
        ckpt = torch.load(gated_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        self.model = GatedFusionNet(
            d_text=int(cfg.get("d_text", 256)),
            d_image=int(cfg.get("d_image", 519)),
            hidden=int(cfg.get("hidden", 192)),
            dropout=float(cfg.get("dropout", 0.25)),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) - 1)))
        self.encoder = ImageEncoder("resnet18")
        if self.encoder.dim != 512:
            raise RuntimeError("ResNet18 不可用，无法生成 519 维图像特征；请检查 torchvision 权重缓存")
        self.image_dim = int(cfg.get("d_image", 519))
        self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def _text_features(self, texts: list[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts)
        z = self.svd.transform(x)
        return self.text_scaler.transform(z).astype(np.float32)

    def _image_features(self, images: list[Image.Image | None]) -> np.ndarray:
        rows = []
        for im in images:
            vec = self.encoder.encode_pil(im).astype(np.float32)
            vec = vec / max(float(np.linalg.norm(vec)), 1e-6)
            st = image_stats(im)
            svec = self.stats_scaler.transform(
                np.array([[st[c] for c in self.stat_cols]], dtype=np.float32))[0] * 0.5
            rows.append(np.concatenate([vec, svec]).astype(np.float32))
        return np.stack(rows)

    def _forward(self, zt: np.ndarray, zi: np.ndarray,
                 mt: np.ndarray, mi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            logits, gate = self.model(
                torch.from_numpy(zt.astype(np.float32)),
                torch.from_numpy(zi.astype(np.float32)),
                torch.from_numpy(mt.astype(np.float32)),
                torch.from_numpy(mi.astype(np.float32)),
            )[:2]
            probs = torch.softmax(logits, dim=1).numpy()
            return probs, gate.numpy()

    @staticmethod
    def _intensity_proxy(probs: np.ndarray) -> int:
        """三类概率映射到 -3~3 的演示强度，不是人工强度标注。"""
        p_neg, _, p_pos = probs
        return int(np.clip(round(3.0 * (float(p_pos) - float(p_neg))), -3, 3))

    def predict(self, text: str, image_path: str | None = None,
                image_base64: str | None = None, explain: bool = True,
                top_k: int = 3, explain_image: bool = True) -> dict:
        t0 = time.time()
        raw = "" if text is None else str(text)
        clean = clean_text(raw)
        if len(clean) < 2:
            raise ValueError("文本为空或过短（清洗后 <2 字）")
        im = _read_image(image_path, image_base64)
        zt = self._text_features([clean])
        if im is None:
            zi = np.zeros((1, self.image_dim), dtype=np.float32)
            mt = np.array([1.0], dtype=np.float32)
            mi = np.array([0.0], dtype=np.float32)
            has_image = False
        else:
            zi = self._image_features([im])
            mt = np.array([1.0], dtype=np.float32)
            mi = np.array([1.0], dtype=np.float32)
            has_image = True
        probs, gate = self._forward(zt, zi, mt, mi)
        p = probs[0]
        idx = int(np.argmax(p))
        label = int(self.classes[idx])
        result = {
            "prediction": {
                "label": label,
                "label_name": LABEL_NAMES.get(label, str(label)),
                "probabilities": {LABEL_NAMES.get(int(c), str(int(c))): round(float(p[i]), 6)
                                  for i, c in enumerate(self.classes)},
                "intensity": self._intensity_proxy(p),
                "intensity_type": "probability_proxy_not_human_annotated",
            },
            "modalities": {
                "text_available": True,
                "image_available": has_image,
                "gate": {"text": round(float(gate[0, 0]), 6),
                         "image": round(float(gate[0, 1]), 6)},
                "main_modality": "文本" if gate[0, 0] >= gate[0, 1] else "图像",
            },
            "input": {"text_clean": clean, "text_len": len(clean),
                      "image_path": image_path or ("base64" if image_base64 else None)},
            "method": "门控融合（文本 TF-IDF+SVD / 图像 ResNet18+统计）",
            "caveats": [
                "情感标签来自公开数据集弱标注，不是人工金标准。",
                "强度是三类概率映射的演示值，不是人工 -3~3 强度标注。",
                "解释是删除式遮挡归因，不是因果证明。",
                "结论只用于聚合舆情辅助，不用于个体心理判断或自动处置。",
            ],
        }
        if explain:
            result["evidence"] = {
                "text_top_tokens": self._text_evidence(clean, zt, zi, mt, mi, idx, top_k),
                "image_grid": self._image_evidence(im, zt, zi) if (has_image and explain_image) else None,
                "method": "文本：删除 jieba 关键词（短文本回退为字符片段）后重算概率；图像：3×3 灰色遮挡后重算概率",
            }
        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return result

    @staticmethod
    def _fallback_spans(clean: str, limit: int = 10) -> list[str]:
        """短文本/关键词不足时的字符片段候选，保证解释卡不会空白。"""
        spans: list[str] = []
        clauses = re.split(r"[\s,，。.!！?？;；:：/、|\\\-—_]+", clean)
        for clause in clauses:
            clause = clause.strip()
            if len(clause) < 2 or not re.search(r"[\u4e00-\u9fffA-Za-z]", clause):
                continue
            cands = []
            if len(clause) <= 10:
                cands.extend(clause[i:i + 2] for i in range(len(clause) - 1))
            else:
                cands.extend(clause[i:i + 3] for i in range(0, min(len(clause) - 2, 15), 2))
            if len(clause) > 4:
                cands.append(clause[:12])
            for c in cands:
                if len(c) >= 2 and c not in spans:
                    spans.append(c)
                if len(spans) >= limit:
                    return spans
        return spans

    def _text_evidence(self, clean: str, zt: np.ndarray, zi: np.ndarray,
                       mt: np.ndarray, mi: np.ndarray, base_idx: int,
                       top_k: int) -> list[dict]:
        toks = []
        for w in tokenize_keywords(clean):
            if len(w) >= 2 and w in clean and w not in toks:
                toks.append(w)
        if len(toks) < 3:
            for span in self._fallback_spans(clean):
                if span not in toks and span in clean:
                    toks.append(span)
                if len(toks) >= 8:
                    break
        toks = toks[:8]
        if not toks:
            return []
        texts = []
        used = []
        for tok in toks:
            new = clean.replace(tok, "")
            if len(new.strip()) < 2 or new == clean:
                continue
            texts.append(new)
            used.append(tok)
        if not texts:
            return []
        zps = self._text_features(texts)
        ips = np.repeat(zi[0:1], len(texts), axis=0)
        mtp = np.repeat(mt[0:1], len(texts), axis=0)
        mip = np.repeat(mi[0:1], len(texts), axis=0)
        pp, _ = self._forward(zps, ips, mtp, mip)
        base_p = self._forward(zt, zi, mt, mi)[0][0, base_idx]
        rows = []
        for tok, new_p in zip(used, pp):
            rows.append({
                "token": tok,
                "prob_drop": round(float(base_p - new_p[base_idx]), 6),
                "pred_after_delete": LABEL_NAMES.get(int(self.classes[int(np.argmax(new_p))]), "—"),
            })
        rows.sort(key=lambda x: x["prob_drop"], reverse=True)
        return rows[:top_k]

    def _image_evidence(self, im: Image.Image, zt: np.ndarray, zi: np.ndarray) -> list[dict]:
        w, h = im.size
        cw, ch = max(1, w // 3), max(1, h // 3)
        imgs = []
        for rr in range(3):
            for cc in range(3):
                im2 = im.copy()
                draw = ImageDraw.Draw(im2)
                x0, y0 = cc * cw, rr * ch
                x1 = w if cc == 2 else (cc + 1) * cw
                y1 = h if rr == 2 else (rr + 1) * ch
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(128, 128, 128))
                imgs.append(im2)
        zi_occ = self._image_features(imgs)
        zt9 = np.repeat(zt[0:1], 9, axis=0)
        mt9 = np.zeros(9, dtype=np.float32)  # 仅图像场景，隔离图像证据
        mi9 = np.ones(9, dtype=np.float32)
        pp, _ = self._forward(zt9, zi_occ, mt9, mi9)
        base_probs = self._forward(zt, zi[0:1], np.zeros(1, np.float32),
                                   np.ones(1, np.float32))[0][0]
        img_idx = int(np.argmax(base_probs))
        base_p = float(base_probs[img_idx])
        rows = []
        for rr in range(3):
            for cc in range(3):
                j = rr * 3 + cc
                rows.append({
                    "row": rr, "col": cc,
                    "prob_drop": round(base_p - float(pp[j, img_idx]), 6),
                })
        return rows


if __name__ == "__main__":
    eng = SentimentInference()
    demo = eng.predict("武汉加油，感谢一线医护人员！", explain=True)
    print(json.dumps(demo, ensure_ascii=False, indent=2))
