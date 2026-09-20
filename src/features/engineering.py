"""Target-free feature engineering for the Spaceship Titanic model matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


SPENDING_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERICAL_FEATURES = ["Age", *SPENDING_COLUMNS, "CabinNumber", "GroupSize", "TotalSpending"]
CATEGORICAL_FEATURES = ["HomePlanet", "CryoSleep", "Destination", "VIP", "CabinDeck", "CabinSide", "HasSpending"]
EXCLUDED_FEATURES = ["PassengerId", "Name", "Cabin", "GroupId"]
REQUIRED_COLUMNS = ["PassengerId", "Cabin", *SPENDING_COLUMNS, "Age", "HomePlanet", "CryoSleep", "Destination", "VIP"]


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derive candidate features without ever reading or accepting the target.

    GroupSize is learned only from the feature rows passed to ``fit``. Unknown
    groups encountered in validation/test are assigned size one, ensuring that
    validation/test rows cannot influence fitted feature statistics.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "TitanicFeatureEngineer":
        self._validate(X)
        group_id = X["PassengerId"].astype("string").str.split("_", n=1).str[0]
        self.group_sizes_ = group_id.value_counts().to_dict()
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "group_sizes_"):
            raise RuntimeError("TitanicFeatureEngineer must be fitted before transformation.")
        self._validate(X)
        features = X.copy()
        cabin = features["Cabin"].astype("string").str.split("/", expand=True)
        features["CabinDeck"] = cabin[0]
        features["CabinNumber"] = pd.to_numeric(cabin[1], errors="coerce")
        features["CabinSide"] = cabin[2]
        features["GroupId"] = features["PassengerId"].astype("string").str.split("_", n=1).str[0]
        features["GroupSize"] = features["GroupId"].map(self.group_sizes_).fillna(1).astype(float)
        features["TotalSpending"] = features[SPENDING_COLUMNS].sum(axis=1, min_count=1)
        features["HasSpending"] = features["TotalSpending"].gt(0).where(features["TotalSpending"].notna())
        categorical = features[CATEGORICAL_FEATURES].astype("object")
        features[CATEGORICAL_FEATURES] = categorical.mask(categorical.isna(), np.nan)
        return features[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    @staticmethod
    def _validate(X: pd.DataFrame) -> None:
        missing = sorted(set(REQUIRED_COLUMNS).difference(X.columns))
        if missing:
            raise ValueError(f"Cannot engineer features; missing columns: {', '.join(missing)}")
        if "Transported" in X.columns:
            raise ValueError("Transported must be removed before feature engineering.")
