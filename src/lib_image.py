# -*- coding: utf-8 -*-
"""图像编码器：ResNet18 预训练特征（默认） + 手工可解释特征（回退/统计）。

- ResNet18 只做冻结特征提取，不微调，保证 CPU 可复现；
- 手工特征用于图像质量统计与遮挡解释的可读性；
- 损坏图片返回零向量并记录，不中断流水线。
"""
from __future__ import annotations

import os
import warnings

import numpy as np
from PIL import Image, ImageOps

warnings.filterwarnings("ignore", category=UserWarning)

IMG_SIZE = 128


def _safe_open(path: str) -> Image.Image | None:
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")
    except Exception:
        return None


def _resize_square(im: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    return im.resize((size, size), Image.BILINEAR)


def handcrafted_features(im: Image.Image | None) -> np.ndarray:
    """96 维手工特征：颜色直方图 + 3x3 网格颜色 + 亮度/饱和度/边缘/熵。"""
    if im is None:
        return np.zeros(96, dtype=np.float32)
    im = _resize_square(im, IMG_SIZE)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    feats = []
    # RGB 直方图 8x3
    for ch in (r, g, b):
        h, _ = np.histogram(ch, bins=8, range=(0, 1), density=True)
        feats.extend(h.tolist())
    # HSV 直方图 8x3
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    for ch in range(3):
        h, _ = np.histogram(hsv[..., ch], bins=8, range=(0, 1), density=True)
        feats.extend(h.tolist())
    # 3x3 网格平均 RGB
    cell = IMG_SIZE // 3
    for i in range(3):
        for j in range(3):
            block = arr[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
            feats.extend(block.reshape(-1, 3).mean(axis=0).tolist())
    # 全局统计
    gy, gx = np.gradient(gray)
    edge = np.sqrt(gx ** 2 + gy ** 2)
    hist32, _ = np.histogram(gray, bins=32, range=(0, 1), density=False)
    p = hist32 / max(hist32.sum(), 1)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                         + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    center = gray[cell:2 * cell, cell:2 * cell].mean()
    border = (gray.sum() - gray[cell:2 * cell, cell:2 * cell].sum()) / max(gray.size - cell * cell, 1)
    stats = [
        float(gray.mean()), float(gray.std()), float(sat.mean()), float(sat.std()),
        float(mx.mean()), float(mx.std()), float(edge.mean()),
        float((edge > 0.15).mean()), entropy, colorfulness,
        float(center - border), float((r > g + 0.08).mean()),
        float((g > r + 0.08).mean()), float((b > r + 0.08).mean()),
        float(im.width / max(im.height, 1)),
    ]
    feats.extend(stats)
    out = np.asarray(feats, dtype=np.float32)
    if out.shape[0] < 96:
        out = np.pad(out, (0, 96 - out.shape[0]))
    return out[:96]


def image_stats(im: Image.Image | None) -> dict:
    if im is None:
        return {"width": 0, "height": 0, "aspect": 0.0, "brightness": 0.0,
                "saturation": 0.0, "contrast": 0.0, "colorfulness": 0.0,
                "edge_density": 0.0, "entropy": 0.0, "warm_ratio": 0.0}
    im2 = _resize_square(im, IMG_SIZE)
    arr = np.asarray(im2, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    gy, gx = np.gradient(gray)
    edge = np.sqrt(gx ** 2 + gy ** 2)
    hist32, _ = np.histogram(gray, bins=32, range=(0, 1), density=False)
    p = hist32 / max(hist32.sum(), 1)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                         + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return {
        "width": int(im.width), "height": int(im.height),
        "aspect": round(float(im.width / max(im.height, 1)), 4),
        "brightness": round(float(gray.mean()), 4),
        "saturation": round(float(sat.mean()), 4),
        "contrast": round(float(gray.std()), 4),
        "colorfulness": round(colorfulness, 4),
        "edge_density": round(float((edge > 0.15).mean()), 4),
        "entropy": round(entropy, 4),
        "warm_ratio": round(float((r > g + 0.08).mean()), 4),
    }


class ImageEncoder:
    """统一图像编码接口。backend='resnet18' 或 'handcrafted'。"""

    def __init__(self, backend: str = "resnet18"):
        self.backend = backend
        self.dim = 512 if backend == "resnet18" else 96
        self._model = None
        self._tf = None
        self._torch = None
        if backend == "resnet18":
            try:
                import torch
                import torchvision
                from torchvision import transforms
                from torchvision.models import ResNet18_Weights, resnet18

                torch.set_num_threads(max(1, min(8, (os.cpu_count() or 4) - 1)))
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                model.fc = torch.nn.Identity()
                model.eval()
                self._model = model
                self._torch = torch
                self._tf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
            except Exception as e:
                print(f"  [WARN] ResNet18 不可用，回退手工特征：{type(e).__name__}: {e}")
                self.backend = "handcrafted"
                self.dim = 96
                self._model = None

    def encode_pil(self, im: Image.Image | None) -> np.ndarray:
        if self.backend == "handcrafted":
            return handcrafted_features(im)
        if im is None:
            return np.zeros(self.dim, dtype=np.float32)
        x = self._tf(im).unsqueeze(0)
        with self._torch.inference_mode():
            v = self._model(x).cpu().numpy().astype(np.float32)[0]
        return v

    def encode_paths(self, paths: list[str], batch_size: int = 64,
                     verbose: bool = True) -> tuple[np.ndarray, list[int]]:
        n = len(paths)
        out = np.zeros((n, self.dim), dtype=np.float32)
        bad = []
        if self.backend == "handcrafted":
            for i, p in enumerate(paths):
                im = _safe_open(p)
                if im is None:
                    bad.append(i)
                out[i] = handcrafted_features(im)
                if verbose and (i + 1) % 1000 == 0:
                    print(f"    图像特征 {i + 1}/{n}")
            return out, bad

        buf_imgs, buf_idx = [], []
        done = 0
        for i, p in enumerate(paths):
            im = _safe_open(p)
            if im is None:
                bad.append(i)
                continue
            buf_imgs.append(im)
            buf_idx.append(i)
            if len(buf_imgs) >= batch_size:
                x = self._torch.stack([self._tf(im) for im in buf_imgs])
                with self._torch.inference_mode():
                    v = self._model(x).cpu().numpy().astype(np.float32)
                out[buf_idx] = v
                done += len(buf_imgs)
                buf_imgs, buf_idx = [], []
                if verbose and done % 1024 < batch_size:
                    print(f"    图像特征 {done}/{n}")
        if buf_imgs:
            x = self._torch.stack([self._tf(im) for im in buf_imgs])
            with self._torch.inference_mode():
                v = self._model(x).cpu().numpy().astype(np.float32)
            out[buf_idx] = v
        return out, bad


def load_image(path: str) -> Image.Image | None:
    return _safe_open(path)
