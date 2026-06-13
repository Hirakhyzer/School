from __future__ import annotations

import argparse

from .model import TopicModel
from .utils import accuracy, load_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_papers.csv")
    parser.add_argument("--model", default="models/topic_classifier.json")
    args = parser.parse_args()

    rows = load_rows(args.data)
    model = TopicModel.train(rows)
    model.save(args.model)

    guesses = [model.predict(row["abstract"]) for row in rows]
    score = accuracy([row["topic"] for row in rows], guesses)
    print(f"Saved model to {args.model}")
    print(f"Rows: {len(rows)}")
    print(f"Accuracy: {score:.3f}")


if __name__ == "__main__":
    main()
