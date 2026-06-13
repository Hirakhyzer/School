# AI Research Topic Classifier

A GitHub-ready machine-learning project that classifies short research paper abstracts into topic areas using a lightweight Naive Bayes text model.

The core model uses only the Python standard library. The uploaded version includes a sample dataset, model code, training script, simple text-labeling command, and a saved model JSON artifact.

## Features

- Trains a multinomial Naive Bayes model from CSV text data.
- Saves the trained model as JSON.
- Labels a new abstract from the command line.
- Includes a sample dataset and a prebuilt model artifact.

## Topics in the sample dataset

- artificial_intelligence
- data_science
- network_systems
- software_engineering

## Quick start

```bash
cd ai-research-topic-classifier
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Build the model artifact from the sample dataset:

```bash
python -m paper_topic_classifier.run_train \
  --data data/sample_papers.csv \
  --model models/topic_classifier.json
```

Label a new abstract:

```bash
python -m paper_topic_classifier.label_text \
  --text "This paper proposes a neural model for classifying technical documents."
```

## Project structure

```text
ai-research-topic-classifier/
├── README.md
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── data/
│   └── sample_papers.csv
├── models/
│   └── topic_classifier.json
└── paper_topic_classifier/
    ├── label_text.py
    ├── model.py
    ├── run_train.py
    └── utils.py
```

## Notes

This is a portfolio and learning project. The sample dataset is intentionally small, so the model is useful for demonstrating the workflow rather than production classification accuracy.
