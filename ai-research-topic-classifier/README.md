# AI Research Topic Classifier

A complete, GitHub-ready machine-learning project that classifies short research paper abstracts into topic areas using a lightweight Naive Bayes text model.

The core model uses only the Python standard library. It includes a sample dataset, training script, prediction CLI, saved model JSON, tests, model card, and GitHub Actions workflow.

## Features

- Trains a multinomial Naive Bayes classifier from CSV text data.
- Saves the trained model as JSON.
- Predicts a topic label for a new abstract from the command line.
- Includes a simple Streamlit UI as an optional interface.
- Includes unit tests and CI.

## Topics in the sample dataset

- artificial_intelligence
- cybersecurity
- data_science
- software_engineering

## Quick start

```bash
cd ai-research-topic-classifier
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Train the model:

```bash
python -m paper_topic_classifier.train \
  --data data/sample_papers.csv \
  --model models/topic_classifier.json
```

Predict a topic:

```bash
python -m paper_topic_classifier.predict \
  --text "This paper proposes a transformer model for classifying technical documents."
```

Run tests:

```bash
python -m unittest discover -s tests
```

Optional Streamlit app:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Project structure

```text
ai-research-topic-classifier/
├── README.md
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── app/
│   └── streamlit_app.py
├── data/
│   └── sample_papers.csv
├── docs/
│   └── model_card.md
├── models/
│   └── topic_classifier.json
├── paper_topic_classifier/
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
└── tests/
    └── test_model.py
```

## Notes

This is a portfolio and learning project. The sample dataset is intentionally small, so the model is useful for demonstrating the workflow rather than production classification accuracy.
