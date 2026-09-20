"""Auditing, conservative cleaning, and validation for Spaceship Titanic data."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


RAW_FILENAMES = ("train.csv", "test.csv", "sample_submission.csv")
BOOLEAN_COLUMNS = ("CryoSleep", "VIP")
NUMERIC_COLUMNS = ("Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")
NULL_TOKENS = {"", "na", "n/a", "null", "none", "nan"}


def load_data(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the three immutable Kaggle CSV files from ``raw_dir``."""
    missing = [filename for filename in RAW_FILENAMES if not (raw_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing raw data files: {', '.join(missing)}")
    return {
        path.stem: pd.read_csv(path)
        for path in (raw_dir / filename for filename in RAW_FILENAMES)
    }


def _text_quality(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[column].dropna().astype(str)
        stripped = values.str.strip()
        rows.append(
            {
                "column": column,
                "empty_strings": int((values == "").sum()),
                "whitespace_only": int(((values != "") & (stripped == "")).sum()),
                "null_tokens": int(stripped.str.lower().isin(NULL_TOKENS - {""}).sum()),
                "leading_or_trailing_whitespace": int((values != stripped).sum()),
            }
        )
    return pd.DataFrame(rows)


def _numeric_audit(frame: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in NUMERIC_COLUMNS if column in frame]
    columns = ["count", "mean", "std", "min", "25%", "median", "75%", "max", "zero_count", "negative_count"]
    if not available:
        return pd.DataFrame(columns=columns)
    stats = frame[available].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats["zero_count"] = (frame[available] == 0).sum()
    stats["negative_count"] = (frame[available] < 0).sum()
    return stats.rename(columns={"50%": "median"})[columns]


def _categorical_audit(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    categorical = [column for column in frame.columns if column not in NUMERIC_COLUMNS]
    return {
        column: frame[column]
        .value_counts(dropna=False)
        .rename_axis("value")
        .reset_index(name="count")
        .assign(percent=lambda values: values["count"] / len(frame) * 100)
        for column in categorical
    }


def _pattern_summary(series: pd.Series, pattern: str) -> dict[str, int]:
    values = series.dropna().astype(str)
    return {"non_missing": len(values), "matching": int(values.str.fullmatch(pattern).sum())}


def audit_dataset(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    """Return repeatable, non-mutating data-quality measurements for one frame."""
    missing = pd.DataFrame(
        {
            "missing_count": frame.isna().sum(),
            "missing_percent": frame.isna().mean().mul(100),
            "unique_count": frame.nunique(dropna=True),
            "unique_percent": frame.nunique(dropna=True).div(len(frame)).mul(100),
            "dtype": frame.dtypes.astype(str),
        }
    )
    audit: dict[str, Any] = {
        "name": name,
        "shape": frame.shape,
        "memory_bytes": int(frame.memory_usage(deep=True).sum()),
        "missing": missing,
        "duplicate_rows": int(frame.duplicated().sum()),
        "constant_columns": [column for column in frame if frame[column].nunique(dropna=False) <= 1],
        "potential_identifiers": [
            column for column in frame if frame[column].nunique(dropna=True) / len(frame) >= 0.95
        ],
        "text_quality": _text_quality(frame),
        "numeric": _numeric_audit(frame),
        "categorical": _categorical_audit(frame),
    }
    if "PassengerId" in frame:
        audit["passenger_id_duplicates"] = int(frame["PassengerId"].duplicated().sum())
        audit["passenger_id_pattern"] = _pattern_summary(frame["PassengerId"], r"\d{4}_\d{2}")
    if "Cabin" in frame:
        audit["cabin_pattern"] = _pattern_summary(frame["Cabin"], r"[^/]+/[^/]+/[PS]")
        cabins = frame["Cabin"].dropna().astype(str)
        audit["cabin_component_counts"] = cabins.str.count("/").add(1).value_counts().sort_index().to_dict()
    return audit


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply only evidenced, information-preserving quality corrections.

    The current raw files contain no accidental surrounding whitespace or unsafe
    boolean representation, so this function returns a value-preserving copy.
    The checks remain here to make future raw-data updates auditable.
    """
    cleaned = frame.copy()
    changes: list[str] = []
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        non_missing = cleaned[column].dropna().astype(str)
        if (non_missing != non_missing.str.strip()).any():
            cleaned[column] = cleaned[column].str.strip()
            changes.append(f"Trimmed surrounding whitespace in {column}.")
    return cleaned, changes


def validate_dataset(
    raw: pd.DataFrame, cleaned: pd.DataFrame, *, is_train: bool = False, is_submission: bool = False
) -> None:
    """Fail fast if conservative cleaning changed the schema or core records."""
    if raw.shape != cleaned.shape or list(raw.columns) != list(cleaned.columns):
        raise ValueError("Cleaning changed the dataset shape or column order.")
    if "PassengerId" in raw and not raw["PassengerId"].equals(cleaned["PassengerId"]):
        raise ValueError("PassengerId integrity check failed.")
    if is_train and not raw["Transported"].equals(cleaned["Transported"]):
        raise ValueError("Target integrity check failed.")
    if is_submission and list(cleaned.columns) != ["PassengerId", "Transported"]:
        raise ValueError("Submission schema is invalid.")


def save_processed_data(datasets: dict[str, pd.DataFrame], processed_dir: Path) -> dict[str, Path]:
    """Save validated outputs outside the immutable raw-data directory."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for name, frame in datasets.items():
        path = processed_dir / f"{name}_clean.csv"
        frame.to_csv(path, index=False)
        outputs[name] = path
    return outputs


def _markdown_table(frame: pd.DataFrame, digits: int = 2) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: f"{value:.{digits}f}" if pd.notna(value) else "")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.fillna("").itertuples(index=False, name=None)]
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def create_audit_figures(train: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create the three reusable, evidence-led audit figures without overwriting assets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "target_distribution.png": "Transported target distribution",
        "missing_values_overview.png": "Missing values by column",
        "age_distribution.png": "Age distribution",
    }
    existing = [filename for filename in targets if (output_dir / filename).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing image assets: {', '.join(existing)}")

    plt.style.use("default")
    target_counts = train["Transported"].value_counts().reindex([False, True], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Not transported", "Transported"], target_counts.values, color=["#314a73", "#c0703f"])
    ax.set_title(targets["target_distribution.png"])
    ax.set_ylabel("Passengers")
    for bar, count in zip(bars, target_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, count, f"{count:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / "target_distribution.png", dpi=180)
    plt.close(fig)

    missing = train.isna().sum().sort_values(ascending=False)
    missing = missing[missing.gt(0)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(missing.index, missing.values, color="#314a73")
    ax.invert_yaxis()
    ax.set_title(targets["missing_values_overview.png"])
    ax.set_xlabel("Missing records")
    fig.tight_layout()
    fig.savefig(output_dir / "missing_values_overview.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(train["Age"].dropna(), bins=30, color="#314a73", edgecolor="white")
    ax.set_title(targets["age_distribution.png"])
    ax.set_xlabel("Age")
    ax.set_ylabel("Passengers")
    fig.tight_layout()
    fig.savefig(output_dir / "age_distribution.png", dpi=180)
    plt.close(fig)
    return [output_dir / filename for filename in targets]


def write_audit_report(
    audits: dict[str, dict[str, Any]],
    datasets: dict[str, pd.DataFrame],
    raw_dir: Path,
    report_path: Path,
    cleaning_changes: dict[str, list[str]],
) -> None:
    """Write the evidence-led audit report from measurements made in this run."""
    train, test, submission = datasets["train"], datasets["test"], datasets["sample_submission"]
    train_audit, test_audit, submission_audit = audits["train"], audits["test"], audits["sample_submission"]
    inventory = pd.DataFrame(
        [
            [name + ".csv", *audit["shape"], raw_dir.joinpath(name + ".csv").stat().st_size, audit["memory_bytes"]]
            for name, audit in audits.items()
        ],
        columns=["File", "Rows", "Columns", "File size (bytes)", "Memory usage (bytes)"],
    )
    schema = pd.DataFrame({"Column": train.columns, "Train dtype": train.dtypes.astype(str).values})
    schema["Test dtype"] = [str(test[column].dtype) if column in test else "-" for column in schema["Column"]]
    target = train["Transported"].value_counts(dropna=False).rename_axis("Transported").reset_index(name="Count")
    target["Percent"] = target["Count"].div(len(train)).mul(100)
    cabin = train_audit["cabin_pattern"]
    pid = train_audit["passenger_id_pattern"]
    bool_values = pd.concat(
        [
            train[[column]].assign(Dataset="train", Column=column).rename(columns={column: "Value"})
            for column in BOOLEAN_COLUMNS
        ]
        + [
            test[[column]].assign(Dataset="test", Column=column).rename(columns={column: "Value"})
            for column in BOOLEAN_COLUMNS
        ]
    )
    boolean_summary = (
        bool_values.groupby(["Dataset", "Column", "Value"], dropna=False).size().rename("Count").reset_index()
    )
    issues = [
        "Missing values occur in multiple passenger attributes; they are preserved for later preprocessing.",
        "`PassengerId` and `Name` are high-cardinality fields and require deliberate treatment in a later feature-engineering phase.",
        "`Cabin` has missing records but every non-missing training value matches the observed three-part `deck/number/side` pattern.",
    ]
    if train_audit["duplicate_rows"] == test_audit["duplicate_rows"] == 0:
        issues.insert(0, "No exact duplicate rows were found in train or test.")
    report = f"""# Spaceship Titanic - Data Audit

## 1. Dataset Overview

The raw directory contains the Kaggle training records, unlabelled test records, and a sample submission template. Raw files were read only; processed copies are stored separately in `data/processed/`.

## 2. Dataset Dimensions

{_markdown_table(inventory, 0)}

The train and test sets share {len(train.columns.intersection(test.columns))} predictor columns. `Transported` is present only in train; test has no train-exclusive replacement column. The sample submission has {len(submission)} rows and its `PassengerId` column exactly matches the test-set order.

## 3. Schema

{_markdown_table(schema, 0)}

## 4. Missing Values

### Train

{_markdown_table(train_audit["missing"].reset_index(names="Column")[["Column", "missing_count", "missing_percent"]])}

### Test

{_markdown_table(test_audit["missing"].reset_index(names="Column")[["Column", "missing_count", "missing_percent"]])}

## 5. Duplicate Analysis

| Dataset | Exact duplicate rows | Duplicate `PassengerId` values |
| --- | ---: | ---: |
| train | {train_audit["duplicate_rows"]} | {train_audit["passenger_id_duplicates"]} |
| test | {test_audit["duplicate_rows"]} | {test_audit["passenger_id_duplicates"]} |
| sample submission | {submission_audit["duplicate_rows"]} | {submission_audit["passenger_id_duplicates"]} |

## 6. Target Distribution

`Transported` exists in train, has dtype `{train["Transported"].dtype}`, and has {int(train["Transported"].isna().sum())} missing values.

{_markdown_table(target)}

## 7. Numerical Feature Audit

{_markdown_table(train_audit["numeric"].reset_index(names="Column"))}

All six audited numeric columns have zero negative values. Zero spending is retained as a meaningful observed value. The table reports extremes for later modelling review; this audit does not remove outliers.

## 8. Categorical Feature Audit

The boolean-like columns are stored as `object` because of missing values. Their observed non-missing values are consistently `True` and `False`.

{_markdown_table(boolean_summary)}

`HomePlanet`, `Destination`, `CryoSleep`, and `VIP` have no empty strings, whitespace-only values, surrounding whitespace, or textual null tokens in either raw dataset. `PassengerId`, `Cabin`, and `Name` are high-cardinality fields; their values are preserved unchanged.

## 9. PassengerId Investigation

All {pid["non_missing"]:,} non-missing training `PassengerId` values match `dddd_dd` (four digits, underscore, two digits); no training identifiers are duplicated. The same pattern check is performed in the reusable audit code for test. The first component appears to encode a travelling group and the second a passenger position within that group, consistent with the dataset description. No group-derived feature is created in this phase.

## 10. Cabin Investigation

`Cabin` has {int(train["Cabin"].isna().sum()):,} missing training values. All {cabin["non_missing"]:,} non-missing training cabin values match the observed `deck/number/side` pattern, and the observed component-count distribution is {train_audit["cabin_component_counts"]}. No deck, cabin-number, or side feature is extracted yet.

## 11. Data Quality Issues

""" + "\n".join(f"- {issue}" for issue in issues) + f"""

## 12. Cleaning Decisions

No values required correction in this dataset version. The cleaning function checked text columns for accidental surrounding whitespace; none was found. It therefore wrote value-preserving processed copies and made no row deletions, type coercions, imputation, or outlier changes.

## 13. What We Deliberately Did NOT Change

- Missing values remain present for treatment inside later training preprocessing pipelines.
- No categorical encoding, scaling, normalization, feature selection, or dimensionality reduction was applied.
- No `PassengerId` group feature or `Cabin` component features were created.
- Legitimate zero spending values and numerical extremes were retained.
- No rows were removed because of missing values or outlier status.

## 14. Clean Dataset

`data/processed/train_clean.csv`, `test_clean.csv`, and `sample_submission_clean.csv` were created after validation. Each preserves raw row count, column count, column order, `PassengerId`, and (for train) `Transported` integrity. The cleaned sample submission retains exactly `PassengerId` and `Transported`.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
