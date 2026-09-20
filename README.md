# Spaceship Titanic — Passenger Transport Prediction

A portfolio-quality machine learning case study based on Kaggle's **Spaceship Titanic** competition. The project investigates passenger records recovered after a collision with a spacetime anomaly and will predict whether each passenger was transported to another dimension.

## The Story

During the voyage of the Spaceship Titanic, the ship collided with a spacetime anomaly. The project turns the recovered passenger records into an evidence-led account of the incident, from data audit to an interactive passenger scanner.

## Problem

This is a binary-classification problem. The target, `Transported`, indicates whether a passenger was transported to another dimension.

## Dataset

The official Kaggle dataset is expected in `data/raw/`:

- `train.csv` — labelled passenger records used for development.
- `test.csv` — passenger records for prediction.
- `sample_submission.csv` — the Kaggle submission format.

Raw data is treated as immutable. Cleaned and derived datasets will be written to `data/processed/`. Dataset files are not ignored by default; whether they are committed should be decided according to their licence, size, and repository policy.

## Machine Learning Pipeline

Data → Cleaning → EDA → Feature Engineering → Preprocessing → Training → Evaluation → Browser Prediction

The final inference approach will be selected only after model evaluation. No backend or API is planned: the future website will remain static and deployable on Netlify.

## Technology

- Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Joblib
- HTML5, CSS3, and Vanilla JavaScript

## Website

The future frontend will be a cinematic, editorial sci-fi data story with custom space illustrations, narrative visualisations, model explanation, and an interactive passenger scanner. It will not use a frontend framework.

## Project Structure

```text
spaceship-titanic-ml/
├── data/
│   ├── raw/                 # Immutable Kaggle source files
│   └── processed/           # Reproducible cleaned/derived data
├── model/                   # Trained model artifacts (created later)
├── notebooks/               # Ordered analysis notebooks
├── src/
│   ├── data/                # Loading and cleaning utilities
│   ├── features/            # Feature engineering utilities
│   ├── models/              # Training and evaluation utilities
│   └── utils/               # Shared helpers
├── web/
│   ├── assets/images/       # Source visual assets
│   ├── assets/generated/    # Generated visual assets
│   ├── css/
│   ├── js/
│   └── index.html           # Static site entry point
├── docs/
├── PROJECT_PLAN.md
├── DATA_DICTIONARY.md
├── EDA_INSIGHTS.md
├── MODEL_RESULTS.md
└── requirements.txt
```

## Development Status

**Under development.** The repository currently contains the project foundation only. No model has been trained, no accuracy has been reported, and deployment has not been completed.

The next phase is **data audit and data cleaning**.
