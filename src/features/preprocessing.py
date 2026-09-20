"""Leakage-safe split and preprocessing assembly for Spaceship Titanic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .engineering import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TitanicFeatureEngineer


def make_preprocessor() -> ColumnTransformer:
    """Build a train-fit-only numerical/categorical preprocessing graph."""
    numerical = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numerical", numerical, NUMERICAL_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )


def make_feature_pipeline() -> Pipeline:
    """Return the reusable engineering + preprocessing pipeline; no estimator is included."""
    return Pipeline([("engineering", TitanicFeatureEngineer()), ("preprocessing", make_preprocessor())])


def split_features_target(train: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42):
    """Create the prescribed stratified holdout from training records only."""
    if "Transported" not in train:
        raise ValueError("Training data requires a Transported target column.")
    X, y = train.drop(columns="Transported"), train["Transported"].astype(int)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def validate_pipeline(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    """Fit solely on training-fold X and verify all required safety properties."""
    X_train, X_valid, y_train, y_valid = split_features_target(train)
    pipeline = make_feature_pipeline()
    train_matrix = pipeline.fit_transform(X_train)
    valid_matrix = pipeline.transform(X_valid)
    test_matrix = pipeline.transform(test)
    unknown_probe = X_valid.copy()
    unknown_probe.loc[unknown_probe.index[:1], "HomePlanet"] = "UnseenPlanet"
    unknown_matrix = pipeline.transform(unknown_probe)
    if "Transported" in X_train.columns or "Transported" in X_valid.columns or "Transported" in test.columns:
        raise AssertionError("Target leaked into model feature inputs.")
    if not all(np.isfinite(matrix).all() for matrix in (train_matrix, valid_matrix, test_matrix, unknown_matrix)):
        raise AssertionError("Preprocessing left non-finite values.")
    if train_matrix.shape[1] != valid_matrix.shape[1] or train_matrix.shape[1] != test_matrix.shape[1]:
        raise AssertionError("Feature dimensions are inconsistent.")
    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_valid": X_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "train_shape": train_matrix.shape,
        "valid_shape": valid_matrix.shape,
        "test_shape": test_matrix.shape,
        "feature_count": train_matrix.shape[1],
        "train_rate": float(y_train.mean()),
        "valid_rate": float(y_valid.mean()),
        "unknown_categories_safe": unknown_matrix.shape == valid_matrix.shape,
    }
