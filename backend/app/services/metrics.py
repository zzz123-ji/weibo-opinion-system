"""人工标注与自动引擎的评估指标。"""

from __future__ import annotations

from collections import Counter, defaultdict

from sqlalchemy.orm import Session

from ..models import SentimentAuto, SentimentLabel
from .sentiment import LABELS


def _manual_majority(db: Session) -> dict[tuple[str, str], str]:
    rows = db.query(SentimentLabel).all()
    grouped: dict[tuple[str, str], list[SentimentLabel]] = defaultdict(list)
    for row in rows:
        grouped[(row.text_type, row.text_id)].append(row)
    result: dict[tuple[str, str], str] = {}
    for key, labels in grouped.items():
        counts = Counter(x.label for x in labels)
        top = max(counts.values())
        winners = [x for x in labels if counts[x.label] == top]
        # 票数打平取最新更新的一条。
        winners.sort(key=lambda x: x.updated_at or x.created_at, reverse=True)
        result[key] = winners[0].label
    return result


def _latest_auto(db: Session) -> dict[tuple[str, str, str], str]:
    """每个 (text, engine) 取最新一条；version 仅用于展示。"""
    rows = (
        db.query(SentimentAuto)
        .order_by(SentimentAuto.created_at.asc(), SentimentAuto.id.asc())
        .all()
    )
    latest: dict[tuple[str, str, str], SentimentAuto] = {}
    for row in rows:
        latest[(row.text_type, row.text_id, row.engine)] = row
    return {k: v.label for k, v in latest.items()}  # type: ignore[return-value]


def evaluate_engines(db: Session) -> dict:
    manual = _manual_majority(db)
    latest = _latest_auto(db)
    engines = sorted({(k[2], v) for k, v in latest.items()})
    # 需要 engine_version：重新按 engine 取最新行的 version。
    version_map: dict[str, str] = {}
    for row in (
        db.query(SentimentAuto)
        .order_by(SentimentAuto.created_at.desc(), SentimentAuto.id.desc())
        .all()
    ):
        version_map.setdefault(row.engine, row.engine_version)

    def f1_for(actual: list[str], pred: list[str], label: str) -> dict:
        tp = sum(1 for a, p in zip(actual, pred) if a == label and p == label)
        fp = sum(1 for a, p in zip(actual, pred) if a != label and p == label)
        fn = sum(1 for a, p in zip(actual, pred) if a == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(1 for a in actual if a == label)
        return {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    result = []
    for engine in sorted({e for e, _ in engines}):
        pairs = []
        for (text_type, text_id), manual_label in manual.items():
            pred = latest.get((text_type, text_id, engine))
            if pred:
                pairs.append((manual_label, pred))
        if not pairs:
            continue
        actual = [a for a, _ in pairs]
        pred = [p for _, p in pairs]
        accuracy = sum(1 for a, p in zip(actual, pred) if a == p) / len(pairs)
        per_class = {label: f1_for(actual, pred, label) for label in LABELS}
        macro_f1 = sum(v["f1"] for v in per_class.values()) / len(LABELS)
        confusion = {a: {p: 0 for p in LABELS} for a in LABELS}
        for a, p in zip(actual, pred):
            confusion[a][p] += 1
        result.append({
            "engine": engine,
            "engine_version": version_map.get(engine, ""),
            "n": len(pairs),
            "accuracy": round(accuracy, 6),
            "macro_f1": round(macro_f1, 6),
            "per_class": per_class,
            "confusion": confusion,
        })
    return {"n_manual_texts": len(manual), "engines": result}
