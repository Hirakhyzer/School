# AI Student Performance Predictor

A compact, portfolio-ready machine-learning project that predicts a student's final score from study habits and prior academic performance.

The core model is implemented with the Python standard library only, so the command-line workflow works without NumPy, pandas, or scikit-learn. A Streamlit app is included as an optional interface.

## Features

- Trains a ridge linear regression model from CSV data.
- Saves the trained model as JSON.
- Predicts final scores from command-line inputs.
- Includes a sample dataset and a pre-trained model artifact.
- Provides a Streamlit dashboard for interactive predictions.
- Includes unit tests and a GitHub Actions workflow.

## Project structure

```text
ai-student-performance-predictor/
├── README.md
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── app/
│   └── streamlit_app.py
├── data/
│   └── sample_students.csv
├── docs/
│   └── model_card.md
├── models/
│   └── student_score_model.json
├── student_performance/
│   ├── __init__.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
└── tests/
    └── test_model.py
```

## Quick start

```bash
cd ai-student-performance-predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

The core training and prediction scripts use only the Python standard library.

## Train the model

```bash
python -m student_performance.train \
  --data data/sample_students.csv \
  --model models/student_score_model.json
```

Example output:

```text
Saved model to models/student_score_model.json
Holdout metrics:
  mae: 0.4098
  rmse: 0.4766
  r2: 0.9985
```

## Make a prediction

```bash
python -m student_performance.predict \
  --hours-studied 6.5 \
  --attendance-rate 0.90 \
  --previous-score 72 \
  --assignments-completed 9
```

## Run tests

```bash
python -m unittest discover -s tests
```

## Optional Streamlit app

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Data fields

| Column | Meaning |
|---|---|
| `student_id` | Student identifier |
| `hours_studied` | Average daily study hours |
| `attendance_rate` | Attendance rate from 0.0 to 1.0 |
| `previous_score` | Previous exam score out of 100 |
| `assignments_completed` | Completed assignments count |
| `final_score` | Final score out of 100 |

## Notes

This project is educational and uses a small sample dataset. For real academic decisions, expand the dataset, audit for bias, validate against unseen cohorts, and avoid using predictions as the sole decision factor.
