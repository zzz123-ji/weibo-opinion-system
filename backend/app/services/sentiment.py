"""情感引擎接口与实现。

设计文档要求引擎可切换、可评估、可重跑。本模块提供三个实现：

- ``local``：优先加载项目已有的文本 TF-IDF + LR 权重；文件缺失时退化为词典模型。
- ``llm``：OpenAI 兼容的 /chat/completions 接口，可配置 base_url、模型名和 key。
- ``mock``：测试与离线演示用，按文本哈希稳定输出。

所有引擎都返回 ``(label, score)``，label 为 pos / neu / neg。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from ..config import settings

logger = logging.getLogger(__name__)

LABELS = ("neg", "neu", "pos")
INT_TO_LABEL = {-1: "neg", 0: "neu", 1: "pos"}
LABEL_TO_INT = {v: k for k, v in INT_TO_LABEL.items()}
CN_LABEL = {"neg": "负面", "neu": "中性", "pos": "正面"}


def normalize_label(value: object) -> str:
    """把 -1/0/1、中文、英文标签统一成 pos/neu/neg。"""
    if isinstance(value, bool):
        return "neu"
    if isinstance(value, (int, float)):
        return INT_TO_LABEL.get(int(value), "neu")
    text = str(value or "").strip().lower()
    mapping = {
        "-1": "neg",
        "0": "neu",
        "1": "pos",
        "neg": "neg",
        "negative": "neg",
        "负面": "neg",
        "负向": "neg",
        "neu": "neu",
        "neutral": "neu",
        "中性": "neu",
        "pos": "pos",
        "positive": "pos",
        "正面": "pos",
        "正向": "pos",
    }
    return mapping.get(text, "neu")


class SentimentEngine(Protocol):
    name: str
    version: str

    def classify(self, texts: list[str]) -> list[tuple[str, float]]:
        ...


class EngineUnavailable(RuntimeError):
    pass


@dataclass
class LexiconEngine:
    """无模型文件时的可解释兜底；不是生产级模型。"""

    name: str = "local"
    version: str = "lexicon-v1"

    POSITIVE = (
        "加油", "感谢", "致敬", "支持", "希望", "爱心", "辛苦", "点赞", "赞",
        "温暖", "感动", "平安", "胜利", "治愈", "好消息", "康复", "谢谢",
    )
    NEGATIVE = (
        "死亡", "去世", "确诊", "感染", "恐慌", "愤怒", "难过", "失望", "悲剧",
        "谣言", "崩溃", "求助", "买不到", "封城", "病毒", "肺炎", "隔离",
        "痛苦", "担心", "害怕", "谴责", "隐瞒",
    )

    def classify(self, texts: list[str]) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        for text in texts:
            pos = sum(1 for w in self.POSITIVE if w in text)
            neg = sum(1 for w in self.NEGATIVE if w in text)
            if pos == neg:
                label, score = "neu", 0.55
            elif pos > neg:
                label, score = "pos", min(0.95, 0.6 + 0.08 * (pos - neg))
            else:
                label, score = "neg", min(0.95, 0.6 + 0.08 * (neg - pos))
            results.append((label, round(score, 4)))
        return results


class LocalTfidfEngine:
    """加载 output/models/baselines.joblib 的文本 LR；失败时回退词典。"""

    name = "local"

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path or settings.local_model_path)
        self._fallback = LexiconEngine()
        self._model = None
        self.version = "lexicon-v1"
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            logger.warning("本地模型不存在，使用词典兜底：%s", self.model_path)
            return
        try:
            import joblib
            import numpy as np  # noqa: F401

            model = joblib.load(self.model_path)
            required = {"vectorizer", "svd", "text_scaler", "text_lr"}
            if not required.issubset(model.keys()):
                logger.warning("本地模型字段不完整，使用词典兜底")
                return
            self._model = model
            self.version = "tfidf-lr-v1"
        except Exception as exc:  # pragma: no cover - 取决于本机模型
            logger.warning("本地模型加载失败，使用词典兜底：%s", exc)

    def classify(self, texts: list[str]) -> list[tuple[str, float]]:
        if self._model is None:
            return self._fallback.classify(texts)
        try:
            import numpy as np

            model = self._model
            x = model["vectorizer"].transform(texts)
            x = model["svd"].transform(x)
            x = model["text_scaler"].transform(x)
            proba = model["text_lr"].predict_proba(x)
            classes = list(model["text_lr"].classes_)
            out: list[tuple[str, float]] = []
            for row in proba:
                idx = int(np.argmax(row))
                out.append((INT_TO_LABEL.get(int(classes[idx]), "neu"), round(float(row[idx]), 6)))
            return out
        except Exception as exc:
            logger.warning("本地模型推理失败，回退词典：%s", exc)
            return self._fallback.classify(texts)


class MockEngine:
    """稳定的测试引擎：按文本哈希给出标签，便于构造可重复数据。"""

    name = "mock"

    def __init__(self, version: str = "mock-v1"):
        self.version = version

    def classify(self, texts: list[str]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for text in texts:
            h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
            label = ("neg", "neu", "pos")[h % 3]
            score = 0.55 + (h % 40) / 100.0
            out.append((label, round(score, 4)))
        return out


class LLMEngine:
    """OpenAI 兼容接口；默认关闭，只有配置了 base_url 和 api_key 才启用。"""

    name = "llm"

    def __init__(self, version: str | None = None):
        self.version = version or settings.llm_model
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        if not self.base_url or not self.api_key:
            raise EngineUnavailable("未配置 LLM_BASE_URL / LLM_API_KEY")

    def _prompt(self, text: str) -> str:
        return (
            "你是微博舆情情感分类器。只输出 JSON，不要解释。\n"
            "标签只能是 pos、neu、neg。score 是 0~1 置信度。\n"
            "示例：\n"
            '文本：武汉加油，感谢一线医护人员！ -> {"label":"pos","score":0.95}\n'
            '文本：今天天气不错 -> {"label":"neu","score":0.8}\n'
            '文本：口罩买不到，太失望了 -> {"label":"neg","score":0.9}\n'
            f"文本：{text}\n输出："
        )

    def _parse(self, content: str) -> tuple[str, float]:
        content = content.strip()
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            raise ValueError(f"LLM 返回不是 JSON：{content[:120]}")
        data = json.loads(match.group(0))
        label = normalize_label(data.get("label"))
        score = float(data.get("score", 0.5))
        return label, max(0.0, min(1.0, score))

    def classify(self, texts: list[str]) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=60) as client:
            for text in texts:
                last_error: Exception | None = None
                for attempt in range(3):
                    try:
                        resp = client.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json={
                                "model": self.model,
                                "temperature": 0,
                                "messages": [{"role": "user", "content": self._prompt(text)}],
                            },
                        )
                        resp.raise_for_status()
                        content = resp.json()["choices"][0]["message"]["content"]
                        results.append(self._parse(content))
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt == 2:
                            raise EngineUnavailable(f"LLM 调用失败：{last_error}") from exc
                else:  # pragma: no cover
                    raise EngineUnavailable("LLM 调用失败")
        return results


def build_engine(name: str = "local", version: str | None = None) -> SentimentEngine:
    name = (name or "local").lower()
    if name == "mock":
        return MockEngine(version or "mock-v1")
    if name == "llm":
        return LLMEngine(version)
    if name == "local":
        return LocalTfidfEngine()
    raise EngineUnavailable(f"未知引擎：{name}")


def available_engines() -> list[str]:
    engines = ["local", "mock"]
    if settings.llm_base_url and settings.llm_api_key:
        engines.insert(0, "llm")
    return engines
