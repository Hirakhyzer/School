"""Utility functions for loading text datasets and scoring predictions."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase text and keep simple word-like tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def load_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def accuracy(actual: Iterable[str], predicted: Iterable[str]) -> float:
    actual_list = list(actual)
    predicted_list = list(predicted)
    if len(actual_list) != len(predicted_list):
        raise ValueError("actual and predicted must have the same length")
    if not actual_list:
        return 0.0
    correct = sum(1 for expected, guess in zip(actual_list, predicted_list) if expected == guess)
    return correct / len(actual_list)
