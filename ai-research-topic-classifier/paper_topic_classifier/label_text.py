from __future__ import annotations

import argparse

from .model import TopicModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/topic_classifier.json")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    model = TopicModel.load(args.model)
    label = model.predict(args.text)
    print(label)


if __name__ == "__main__":
    main()
