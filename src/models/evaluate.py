"""Focused Step 05 evaluation and production-artifact assembly."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.features.engineering import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.features.preprocessing import split_features_target
from .train import RANDOM_STATE, baseline_models, make_model_pipeline


STEP4_VALIDATION = pd.DataFrame(
    [
        ["Logistic Regression", .7901, .7880, .7979, .7930, .8835],
        ["Decision Tree", .7591, .7654, .7523, .7588, .7595],
        ["Random Forest", .8120, .8385, .7763, .8062, .8952],
        ["Gradient Boosting", .8016, .7940, .8185, .8061, .9039],
    ], columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
)
STEP4_CV = pd.DataFrame(
    [
        ["Logistic Regression", .7893, .0035, .7953, .0043, .8779, .0015],
        ["Decision Tree", .7409, .0135, .7424, .0127, .7410, .0135],
        ["Random Forest", .8001, .0057, .7940, .0059, .8811, .0042],
        ["Gradient Boosting", .8064, .0070, .8122, .0067, .8933, .0029],
    ], columns=["Model", "Accuracy mean", "Accuracy std", "F1 mean", "F1 std", "ROC-AUC mean", "ROC-AUC std"],
)


def evaluate_candidate(train: pd.DataFrame) -> dict[str, object]:
    """Fit the selected configuration on the existing training fold only."""
    X_train, X_valid, y_train, y_valid = split_features_target(train)
    pipeline = make_model_pipeline(baseline_models()["Gradient Boosting"])
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_valid)
    probabilities = pipeline.predict_proba(X_valid)[:, 1]
    engineered = pipeline.named_steps["engineering"].transform(X_valid)
    errors = engineered.copy()
    errors["Actual"] = y_valid.to_numpy()
    errors["Predicted"] = predicted
    errors["Probability"] = probabilities
    errors["Error"] = errors.Actual.ne(errors.Predicted)
    errors["AgeGroup"] = pd.cut(errors.Age, [-1, 12, 17, 29, 44, 64, 100], labels=["0-12", "13-17", "18-29", "30-44", "45-64", "65+"]).astype("object").fillna("Missing")
    errors["CryoSleep"] = errors.CryoSleep.astype("object").where(errors.CryoSleep.notna(), "Missing")
    error_rates = {
        "age": _error_summary(errors, "AgeGroup"),
        "cryo": _error_summary(errors, "CryoSleep"),
        "homeplanet": _error_summary(errors, "HomePlanet"),
        "destination": _error_summary(errors, "Destination"),
        "deck": _error_summary(errors, "CabinDeck"),
        "spending": _error_summary(errors, "HasSpending"),
        "group": _error_summary(errors, "GroupSize"),
    }
    matrix = confusion_matrix(y_valid, predicted)
    metrics = {
        "Accuracy": accuracy_score(y_valid, predicted), "Precision": precision_score(y_valid, predicted),
        "Recall": recall_score(y_valid, predicted), "F1": f1_score(y_valid, predicted),
        "ROC-AUC": roc_auc_score(y_valid, probabilities),
    }
    return {"pipeline": pipeline, "X_train": X_train, "X_valid": X_valid, "y_valid": y_valid, "metrics": metrics, "matrix": matrix, "errors": errors, "error_rates": error_rates}


def _error_summary(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    labels = frame[column].astype("object").where(frame[column].notna(), "Missing")
    return frame.assign(_group=labels).groupby("_group", dropna=False).Error.agg(records="size", errors="sum", error_rate="mean").reset_index(names=column).sort_values("error_rate", ascending=False)


def _save(fig: plt.Figure, path: Path) -> None:
    if path.exists():
        plt.close(fig)
        return
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def create_evaluation_charts(result: dict[str, object], output_dir: Path) -> list[str]:
    """Create focused evaluation charts without overwriting previous assets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    matrix: np.ndarray = result["matrix"]  # type: ignore[assignment]
    fig, ax = plt.subplots(figsize=(5, 4.5)); image = ax.imshow(matrix, cmap="Blues")
    ax.set(title="Gradient Boosting validation confusion matrix", xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1], xticklabels=["False", "True"], yticklabels=["False", "True"])
    for (row, col), value in np.ndenumerate(matrix): ax.text(col, row, str(value), ha="center", va="center", fontsize=12)
    fig.colorbar(image, ax=ax); _save(fig, output_dir / "final_confusion_matrix.png"); names.append("final_confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(8, 4.8)); values = STEP4_VALIDATION.set_index("Model")[["Precision", "Recall"]]
    values.plot.bar(ax=ax, color=["#304a6e", "#c46d3b"]); ax.set(title="Validation precision and recall", ylabel="Score", ylim=(0, 1)); ax.legend(loc="lower right")
    _save(fig, output_dir / "precision_recall_comparison.png"); names.append("precision_recall_comparison.png")
    fig, ax = plt.subplots(figsize=(8, 4.8)); cv = STEP4_CV.set_index("Model")
    ax.errorbar(range(len(cv)), cv["ROC-AUC mean"], yerr=cv["ROC-AUC std"], fmt="o", capsize=5, color="#304a6e")
    ax.set(title="Five-fold ROC-AUC stability", ylabel="Mean ROC-AUC ± standard deviation", xticks=range(len(cv)), xticklabels=cv.index, ylim=(.70, .92)); ax.tick_params(axis="x", rotation=20)
    _save(fig, output_dir / "cv_stability.png"); names.append("cv_stability.png")
    for key, filename, title in [("age", "error_rate_by_age_group.png", "Validation error rate by age group"), ("cryo", "error_rate_by_cryo_sleep.png", "Validation error rate by CryoSleep")]:
        rates: pd.DataFrame = result["error_rates"][key]  # type: ignore[index]
        fig, ax = plt.subplots(figsize=(7, 4.5)); bars = ax.bar(rates.iloc[:, 0].astype(str), rates.error_rate * 100, color="#c46d3b")
        ax.set(title=title, ylabel="Error rate (%)", xlabel=rates.columns[0], ylim=(0, max(30, rates.error_rate.max() * 115)))
        for bar, rate in zip(bars, rates.error_rate): ax.text(bar.get_x()+bar.get_width()/2, rate*100+.7, f"{rate:.1%}", ha="center", fontsize=8)
        _save(fig, output_dir / filename); names.append(filename)
    return names


def save_production_artifact(train: pd.DataFrame, model_dir: Path) -> tuple[Path, Path]:
    """Refit the selected fixed configuration on all labelled records and save once."""
    X = train.drop(columns="Transported")
    y = train["Transported"].astype(int)
    pipeline = make_model_pipeline(baseline_models()["Gradient Boosting"])
    pipeline.fit(X, y)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact = model_dir / "spaceship_titanic_pipeline.joblib"
    joblib.dump(pipeline, artifact)
    metadata = {
        "model_name": "GradientBoostingClassifier", "target": "Transported", "random_state": RANDOM_STATE,
        "training_rows": len(train), "validation_split": {"test_size": .2, "stratify": "Transported", "random_state": RANDOM_STATE},
        "numerical_features": NUMERICAL_FEATURES, "categorical_features": CATEGORICAL_FEATURES,
        "engineered_features": ["CabinDeck", "CabinNumber", "CabinSide", "GroupSize", "TotalSpending", "HasSpending"],
        "preprocessing": "Numerical median imputation + StandardScaler; categorical most-frequent imputation + OneHotEncoder(handle_unknown='ignore').",
        "artifact_version": "1.0", "created_date": date.today().isoformat(),
    }
    metadata_path = model_dir / "feature_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return artifact, metadata_path


def _table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number"): display[column] = display[column].map(lambda x: f"{x:.4f}")
    return "\n".join(["| " + " | ".join(map(str, display.columns)) + " |", "| " + " | ".join(["---"] * len(display.columns)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None)])


def append_final_evaluation(result: dict[str, object], path: Path, artifact: Path) -> None:
    """Append the final, measured evaluation to the existing Step 04 report."""
    text = path.read_text(encoding="utf-8")
    marker = "\n## Final Evaluation\n"
    if marker in text: text = text.split(marker, 1)[0].rstrip() + "\n"
    metrics = result["metrics"]
    matrix: np.ndarray = result["matrix"]  # type: ignore[assignment]
    age: pd.DataFrame = result["error_rates"]["age"]  # type: ignore[index]
    cryo: pd.DataFrame = result["error_rates"]["cryo"]  # type: ignore[index]
    text += f"""

## Final Evaluation

The selected Gradient Boosting configuration was re-fit on the original 6,954-row training fold and evaluated on the unchanged 1,739-row validation fold. At the default 0.50 probability threshold: accuracy={metrics['Accuracy']:.4f}, precision={metrics['Precision']:.4f}, recall={metrics['Recall']:.4f}, F1={metrics['F1']:.4f}, ROC-AUC={metrics['ROC-AUC']:.4f}. ROC-AUC measures ranking discrimination over thresholds; this single holdout remains only one sample of future performance.

Its validation confusion matrix is TN={matrix[0,0]}, FP={matrix[0,1]}, FN={matrix[1,0]}, TP={matrix[1,1]}; `final_confusion_matrix.png` records the same values. `precision_recall_comparison.png` and `cv_stability.png` summarize the fixed Step 04 comparisons.

## Error Analysis

Error rates are descriptive validation-set summaries, not causes of error. Age-group results:

{_table(age[["AgeGroup", "records", "errors", "error_rate"]])}

CryoSleep results:

{_table(cryo[["CryoSleep", "records", "errors", "error_rate"]])}

The dedicated charts show these subgroup rates. Small groups should be interpreted cautiously.

## Production Candidate

Gradient Boosting is the production candidate configuration. Its fixed holdout ROC-AUC (0.9039) and five-fold mean ROC-AUC (0.8933 ± 0.0029) are the highest recorded; its five-fold mean F1 is also the highest recorded (0.8122 ± 0.0067). This is an engineering choice from the measured baselines, not a guarantee of production performance or a causal conclusion.

## Final Artifact

The final pipeline, including feature engineering, preprocessing, and GradientBoostingClassifier, was re-fit on all 8,693 labelled processed rows and saved to `{artifact.as_posix()}`. It accepts raw passenger feature columns (without `Transported`) and supports `predict` and `predict_proba`.
"""
    path.write_text(text, encoding="utf-8")


def write_model_card(result: dict[str, object], path: Path, artifact: Path) -> None:
    metrics = result["metrics"]; matrix: np.ndarray = result["matrix"]  # type: ignore[assignment]
    path.write_text(f"""# Spaceship Titanic Model Card

## Model

GradientBoostingClassifier (`random_state=42`) inside one scikit-learn pipeline.

## Intended Use

Predict the binary `Transported` outcome for passenger records matching the project schema. This is a portfolio/competition model, not a decision-support system.

## Input Features

Original passenger fields excluding `Transported`; `PassengerId`, `Name`, and `Cabin` are consumed only through documented engineering or excluded after transformation.

## Feature Engineering

Cabin deck/number/side, training-fit GroupSize, TotalSpending, and HasSpending are derived without target aggregation.

## Preprocessing

Numerical fields use train-fit median imputation and StandardScaler. Categorical fields use train-fit most-frequent imputation and OneHotEncoder(handle_unknown='ignore').

## Training Data

The final artifact was fit on all 8,693 rows of `data/processed/train_clean.csv` after candidate selection.

## Validation Strategy

Selection used a stratified 80/20 holdout (`random_state=42`) and fixed five-fold stratified CV on the training fold. Kaggle test records were not used for selection.

## Metrics

On the validation split: accuracy {metrics['Accuracy']:.4f}, precision {metrics['Precision']:.4f}, recall {metrics['Recall']:.4f}, F1 {metrics['F1']:.4f}, ROC-AUC {metrics['ROC-AUC']:.4f}. Confusion matrix: TN={matrix[0,0]}, FP={matrix[0,1]}, FN={matrix[1,0]}, TP={matrix[1,1]}.

## Error Analysis

Validation error rates were summarized by age group and CryoSleep. They are descriptive subgroup summaries and should not be interpreted as causal.

## Limitations

Metrics are tied to this project split and do not guarantee production performance. Missingness, subgroup sample size, train/test distribution changes, and competition-specific data limits remain relevant.

## Reproducibility

Use Python 3.12 with project dependencies and load `{artifact.as_posix()}` with joblib. The metadata file records feature configuration and split settings.

## Deployment Notes

The artifact requires the same input column schema and compatible package versions. No API or browser deployment is created in this phase.
""", encoding="utf-8")
