"""Reproducible baseline training for Spaceship Titanic."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.features.engineering import TitanicFeatureEngineer
from src.features.preprocessing import make_preprocessor, split_features_target


RANDOM_STATE = 42


def baseline_models() -> dict[str, object]:
    """Return small, fixed baseline configurations rather than tuned models."""
    return {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def make_model_pipeline(model: object) -> Pipeline:
    """Give every classifier an independent copy of the same feature pipeline."""
    return Pipeline(
        [
            ("engineering", TitanicFeatureEngineer()),
            ("preprocessing", make_preprocessor()),
            ("model", clone(model)),
        ]
    )


def run_experiments(train: pd.DataFrame) -> dict[str, object]:
    """Train baselines on one holdout and estimate their 5-fold CV variation."""
    X_train, X_valid, y_train, y_valid = split_features_target(train)
    models = baseline_models()
    fitted: dict[str, Pipeline] = {}
    metrics: list[dict[str, float | str]] = []
    matrices: dict[str, np.ndarray] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for name, model in models.items():
        pipeline = make_model_pipeline(model)
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_valid)
        probabilities = pipeline.predict_proba(X_valid)[:, 1]
        metrics.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_valid, predicted),
                "Precision": precision_score(y_valid, predicted, zero_division=0),
                "Recall": recall_score(y_valid, predicted, zero_division=0),
                "F1": f1_score(y_valid, predicted, zero_division=0),
                "ROC-AUC": roc_auc_score(y_valid, probabilities),
            }
        )
        matrices[name] = confusion_matrix(y_valid, predicted)
        fpr, tpr, _ = roc_curve(y_valid, probabilities)
        curves[name] = (fpr, tpr, auc(fpr, tpr))
        fitted[name] = pipeline
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        scores = cross_validate(
            make_model_pipeline(model), X_train, y_train, cv=cv,
            scoring={"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}, n_jobs=1,
        )
        cv_rows.append(
            {
                "Model": name,
                "Accuracy mean": scores["test_accuracy"].mean(), "Accuracy std": scores["test_accuracy"].std(),
                "F1 mean": scores["test_f1"].mean(), "F1 std": scores["test_f1"].std(),
                "ROC-AUC mean": scores["test_roc_auc"].mean(), "ROC-AUC std": scores["test_roc_auc"].std(),
            }
        )
    return {
        "validation": pd.DataFrame(metrics), "cross_validation": pd.DataFrame(cv_rows),
        "matrices": matrices, "curves": curves, "fitted": fitted,
        "split": (len(X_train), len(X_valid)), "target_rates": (float(y_train.mean()), float(y_valid.mean())),
    }


def _save(fig: plt.Figure, path: Path) -> bool:
    if path.exists():
        plt.close(fig)
        return False
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    return True


def create_model_charts(results: dict[str, object], output_dir: Path) -> list[str]:
    """Create comparison, confusion-matrix, ROC, and RF-importance charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    validation: pd.DataFrame = results["validation"]  # type: ignore[assignment]
    fig, ax = plt.subplots(figsize=(9, 5))
    shown = validation.set_index("Model")[["Accuracy", "F1", "ROC-AUC"]]
    shown.plot.bar(ax=ax, color=["#304a6e", "#c46d3b", "#6d8b74"])
    ax.set(title="Baseline validation metric comparison", ylabel="Score", ylim=(0, 1)); ax.legend(loc="lower right")
    if _save(fig, output_dir / "model_comparison.png"): created.append("model_comparison.png")
    for name, matrix in results["matrices"].items():  # type: ignore[union-attr]
        fig, ax = plt.subplots(figsize=(4.5, 4))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set(title=f"{name} confusion matrix", xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1], xticklabels=["False", "True"], yticklabels=["False", "True"])
        for (row, col), value in np.ndenumerate(matrix): ax.text(col, row, str(value), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        filename = f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
        if _save(fig, output_dir / filename): created.append(filename)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (fpr, tpr, score) in results["curves"].items():  # type: ignore[union-attr]
        ax.plot(fpr, tpr, label=f"{name} ({score:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1); ax.set(title="Validation ROC curves", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1)); ax.legend(loc="lower right")
    if _save(fig, output_dir / "roc_curve_comparison.png"): created.append("roc_curve_comparison.png")
    pipeline: Pipeline = results["fitted"]["Random Forest"]  # type: ignore[index]
    names = pipeline.named_steps["preprocessing"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    ranked = pd.Series(importances, index=names).sort_values(ascending=False).head(15).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6)); ax.barh(ranked.index, ranked.values, color="#304a6e")
    ax.set(title="Random Forest model feature importance", xlabel="Internal importance")
    if _save(fig, output_dir / "feature_importance.png"): created.append("feature_importance.png")
    return created


def _markdown(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number"): display[column] = display[column].map(lambda x: f"{x:.4f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def write_results(results: dict[str, object], path: Path) -> None:
    """Write measured results, without selecting a final production model."""
    validation: pd.DataFrame = results["validation"]  # type: ignore[assignment]
    cross_validation: pd.DataFrame = results["cross_validation"]  # type: ignore[assignment]
    confusion = "\n".join(
        f"- **{name}:** TN={matrix[0,0]}, FP={matrix[0,1]}, FN={matrix[1,0]}, TP={matrix[1,1]}"
        for name, matrix in results["matrices"].items()  # type: ignore[union-attr]
    )
    report = f"""# Spaceship Titanic - Model Results

## 1. Experiment Setup

Experiments use `data/processed/train_clean.csv`, target `Transported`, and a stratified 80/20 split with `random_state=42` ({results['split'][0]:,} train / {results['split'][1]:,} validation). Every model uses the same target-free feature engineer followed by median-impute/scale numerical preprocessing and most-frequent-impute/one-hot categorical preprocessing. Kaggle test data is not used for model selection or metrics.

## 2. Models Tested

Logistic Regression (`max_iter=2000`), Decision Tree, Random Forest (200 estimators), and Gradient Boosting; all use `random_state=42`.

## 3. Validation Results

{_markdown(validation)}

## 4. Cross-Validation Results

Five-fold stratified cross-validation (`shuffle=True`, `random_state=42`) was run on the training-fold records. Values are mean and standard deviation across folds.

{_markdown(cross_validation)}

## 5. Confusion Matrices

{confusion}

## 6. ROC-AUC

Validation ROC-AUC measurements are in the validation table and all four curves are shown in `roc_curve_comparison.png`.

## 7. Feature Importance

`feature_importance.png` shows Random Forest internal impurity-based feature importance after preprocessing. It describes that fitted model's split criterion usage; it is not causal importance.

## 8. Observations

The tables report holdout and cross-validation measurements for fixed baselines. Differences should be interpreted alongside fold variation rather than as a final selection decision.

## 9. Model Selection Considerations

Step 05 should consider validation results, cross-validation stability, interpretability, inference complexity, and model size before selecting a final artifact. No final model is selected here.
"""
    path.write_text(report, encoding="utf-8")
