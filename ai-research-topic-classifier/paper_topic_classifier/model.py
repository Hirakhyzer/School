"""A tiny multinomial Naive Bayes text model."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from .utils import tokenize


class TopicModel:
    def __init__(self, labels, label_counts, token_counts, total_tokens, vocabulary, alpha=1.0):
        self.labels = list(labels)
        self.label_counts = dict(label_counts)
        self.token_counts = {label: dict(counts) for label, counts in token_counts.items()}
        self.total_tokens = dict(total_tokens)
        self.vocabulary = set(vocabulary)
        self.alpha = float(alpha)

    @classmethod
    def train(cls, rows: list[Mapping[str, str]], text_field: str = "abstract", label_field: str = "topic", alpha: float = 1.0):
        label_counts = Counter()
        token_counts = defaultdict(Counter)
        total_tokens = Counter()
        vocabulary = set()

        for row in rows:
            label = row[label_field]
            tokens = tokenize(row[text_field])
            label_counts[label] += 1
            token_counts[label].update(tokens)
            total_tokens[label] += len(tokens)
            vocabulary.update(tokens)

        return cls(
            labels=sorted(label_counts),
            label_counts=label_counts,
            token_counts=token_counts,
            total_tokens=total_tokens,
            vocabulary=vocabulary,
            alpha=alpha,
        )

    def predict(self, text: str) -> str:
        scores = self.predict_proba(text)
        return max(scores, key=scores.get)

    def predict_proba(self, text: str) -> dict[str, float]:
        tokens = tokenize(text)
        total_docs = sum(self.label_counts.values())
        vocab_size = max(1, len(self.vocabulary))
        log_scores = {}

        for label in self.labels:
            log_prob = math.log(self.label_counts[label] / total_docs)
            denominator = self.total_tokens.get(label, 0) + self.alpha * vocab_size
            counts = self.token_counts.get(label, {})
            for token in tokens:
                token_count = counts.get(token, 0)
                log_prob += math.log((token_count + self.alpha) / denominator)
            log_scores[label] = log_prob

        max_log = max(log_scores.values())
        exp_scores = {label: math.exp(value - max_log) for label, value in log_scores.items()}
        total = sum(exp_scores.values())
        return {label: value / total for label, value in exp_scores.items()}

    def to_dict(self):
        return {
            "labels": self.labels,
            "label_counts": self.label_counts,
            "token_counts": self.token_counts,
            "total_tokens": self.total_tokens,
            "vocabulary": sorted(self.vocabulary),
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            labels=payload["labels"],
            label_counts=payload["label_counts"],
            token_counts=payload["token_counts"],
            total_tokens=payload["total_tokens"],
            vocabulary=payload["vocabulary"],
            alpha=payload.get("alpha", 1.0),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
