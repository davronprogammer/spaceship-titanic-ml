"""Export the fitted Spaceship Titanic deployment pipeline to ONNX.

The saved sklearn pipeline includes a project-specific feature engineer, which
skl2onnx cannot serialize. This exporter preserves the fitted preprocessing and
GradientBoostingClassifier in ONNX. Browser clients provide the exact engineered
contract (CabinDeck, CabinNumber, CabinSide, GroupSize, TotalSpending,
HasSpending); Cabin parsing is documented in feature metadata.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"
ARTIFACT = MODEL_DIR / "spaceship_titanic_pipeline.joblib"
ONNX_ARTIFACT = MODEL_DIR / "spaceship_titanic_pipeline.onnx"
METADATA = MODEL_DIR / "feature_metadata.json"

NUMERICAL = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "CabinNumber", "GroupSize", "TotalSpending"]
CATEGORICAL = ["HomePlanet", "CryoSleep", "Destination", "VIP", "CabinDeck", "CabinSide", "HasSpending"]
SPENDING = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def browser_contract_metadata(categorical_fill_values: dict[str, str]) -> dict[str, object]:
    return {
        "artifact": ONNX_ARTIFACT.name,
        "input_contract": {
            "numerical": NUMERICAL,
            "categorical": CATEGORICAL,
            "onnx_input_types": {**{name: "float32" for name in NUMERICAL}, **{name: "string" for name in CATEGORICAL}},
            "browser_derived_features": {
                "CabinDeck": "First segment of Cabin split on '/'.",
                "CabinNumber": "Numeric second segment of Cabin split on '/'; missing/malformed becomes missing.",
                "CabinSide": "Third segment of Cabin split on '/'.",
                "TotalSpending": "Sum of RoomService, FoodCourt, ShoppingMall, Spa, and VRDeck when at least one is supplied.",
                "HasSpending": "String boolean of TotalSpending > 0; missing when TotalSpending is missing.",
                "GroupSize": "Explicit browser field. PassengerId is intentionally not collected, so group membership cannot be inferred client-side.",
            },
            "categorical_missing_replacements": categorical_fill_values,
        },
        "class_labels": [False, True],
        "class_order": [0, 1],
        "outputs": {"label": "label", "probabilities": "probabilities"},
        "conversion_note": "ONNX contains the fitted ColumnTransformer and GradientBoostingClassifier. The Python-only TitanicFeatureEngineer is represented by the documented browser input contract.",
    }


def make_onnx_inputs() -> list[tuple[str, object]]:
    return [(name, FloatTensorType([None, 1])) for name in NUMERICAL] + [(name, StringTensorType([None, 1])) for name in CATEGORICAL]


def engineered_from_raw(pipeline: object, raw: pd.DataFrame) -> pd.DataFrame:
    return pipeline.named_steps["engineering"].transform(raw)


def as_onnx_inputs(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    feeds: dict[str, np.ndarray] = {}
    for name in NUMERICAL:
        feeds[name] = frame[[name]].astype("float32").to_numpy()
    for name in CATEGORICAL:
        feeds[name] = frame[[name]].astype(str).to_numpy()
    return feeds


def make_browser_deploy_pipeline(original: Pipeline) -> tuple[Pipeline, dict[str, str]]:
    """Remove only the unsupported string imputer after applying its fitted values."""
    preprocessing = original.named_steps["preprocessing"]
    numerical_pipeline = deepcopy(preprocessing.named_transformers_["numerical"])
    categorical_pipeline = preprocessing.named_transformers_["categorical"]
    categorical_imputer = categorical_pipeline.named_steps["imputer"]
    categorical_encoder = deepcopy(categorical_pipeline.named_steps["encoder"])
    # Browser ONNX inputs use strings. Preserve each fitted category and its
    # ordering while representing boolean categories as their string values.
    categorical_encoder.categories_ = [np.asarray([str(value) for value in categories], dtype=object) for categories in categorical_encoder.categories_]
    replacements = {name: str(value) for name, value in zip(CATEGORICAL, categorical_imputer.statistics_, strict=True)}
    # Retain the fitted ColumnTransformer internals rather than fitting a new
    # object. Only replace its categorical imputer+encoder branch with the
    # already-fitted encoder after browser-side missing-value replacement.
    browser_preprocessing = deepcopy(preprocessing)
    browser_transformers = [("numerical", numerical_pipeline, NUMERICAL), ("categorical", categorical_encoder, CATEGORICAL)]
    browser_preprocessing.transformers = browser_transformers
    browser_preprocessing.transformers_ = browser_transformers
    return Pipeline([("preprocessing", browser_preprocessing), ("model", original.named_steps["model"])]), replacements


def apply_categorical_replacements(frame: pd.DataFrame, replacements: dict[str, str]) -> pd.DataFrame:
    result = frame.copy()
    for name, value in replacements.items():
        result[name] = result[name].fillna(value).map(str)
    return result


def main() -> None:
    original = joblib.load(ARTIFACT)
    print(f"Pipeline: {type(original).__name__}")
    print(f"Steps: {list(original.named_steps)}")
    print(f"Classifier: {type(original.named_steps['model']).__name__}; classes={original.named_steps['model'].classes_.tolist()}")

    deploy_pipeline, categorical_replacements = make_browser_deploy_pipeline(original)
    onnx_model = to_onnx(
        deploy_pipeline,
        initial_types=make_onnx_inputs(),
        target_opset={"": 18, "ai.onnx.ml": 3},
        options={id(deploy_pipeline.named_steps["model"]): {"zipmap": False}},
    )
    ONNX_ARTIFACT.write_bytes(onnx_model.SerializeToString())

    raw = pd.read_csv(ROOT / "data" / "raw" / "test.csv").head(5)
    engineered = apply_categorical_replacements(engineered_from_raw(original, raw), categorical_replacements)
    original_labels = original.predict(raw)
    original_probabilities = original.predict_proba(raw)[:, 1]
    sklearn_labels = deploy_pipeline.predict(engineered)
    sklearn_probabilities = deploy_pipeline.predict_proba(engineered)[:, 1]

    session = ort.InferenceSession(ONNX_ARTIFACT.as_posix(), providers=["CPUExecutionProvider"])
    print("ONNX inputs:", [(item.name, item.type) for item in session.get_inputs()])
    print("ONNX outputs:", [(item.name, item.type) for item in session.get_outputs()])
    outputs = session.run(None, as_onnx_inputs(engineered))
    label_output = np.asarray(outputs[0]).reshape(-1).astype(int)
    probability_output = outputs[1]
    if isinstance(probability_output, list):
        onnx_probabilities = np.array([row[1] for row in probability_output], dtype=float)
    else:
        onnx_probabilities = np.asarray(probability_output)[:, 1]
    differences = np.abs(sklearn_probabilities - onnx_probabilities)
    if not np.array_equal(original_labels, sklearn_labels) or not np.allclose(original_probabilities, sklearn_probabilities, atol=1e-12):
        raise RuntimeError("Browser deployment preprocessing no longer matches the canonical joblib pipeline.")
    for index, (sk_label, onnx_label, sk_prob, onnx_prob, delta) in enumerate(zip(sklearn_labels, label_output, sklearn_probabilities, onnx_probabilities, differences, strict=True), start=1):
        print(f"sample={index} joblib={int(sk_label)} onnx={int(onnx_label)} joblib_p={sk_prob:.10f} onnx_p={onnx_prob:.10f} delta={delta:.3e}")
    if not np.array_equal(sklearn_labels.astype(int), label_output) or float(differences.max()) > 1e-6:
        raise RuntimeError("ONNX validation failed: predictions do not match the fitted sklearn deployment pipeline.")

    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    metadata["browser_inference"] = browser_contract_metadata(categorical_replacements)
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {ONNX_ARTIFACT.relative_to(ROOT)}; validation passed for 5 samples.")


if __name__ == "__main__":
    main()
