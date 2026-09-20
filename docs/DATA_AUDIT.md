# Spaceship Titanic - Data Audit

## 1. Dataset Overview

The raw directory contains the Kaggle training records, unlabelled test records, and a sample submission template. Raw files were read only; processed copies are stored separately in `data/processed/`.

## 2. Dataset Dimensions

| File | Rows | Columns | File size (bytes) | Memory usage (bytes) |
| --- | --- | --- | --- | --- |
| train.csv | 8693 | 14 | 805421 | 3544706 |
| test.csv | 4277 | 13 | 372487 | 1739724 |
| sample_submission.csv | 4277 | 2 | 59902 | 243921 |

The train and test sets share 13 predictor columns. `Transported` is present only in train; test has no train-exclusive replacement column. The sample submission has 4277 rows and its `PassengerId` column exactly matches the test-set order.

## 3. Schema

| Column | Train dtype | Test dtype |
| --- | --- | --- |
| PassengerId | str | str |
| HomePlanet | str | str |
| CryoSleep | object | object |
| Cabin | str | str |
| Destination | str | str |
| Age | float64 | float64 |
| VIP | object | object |
| RoomService | float64 | float64 |
| FoodCourt | float64 | float64 |
| ShoppingMall | float64 | float64 |
| Spa | float64 | float64 |
| VRDeck | float64 | float64 |
| Name | str | str |
| Transported | bool | - |

## 4. Missing Values

### Train

| Column | missing_count | missing_percent |
| --- | --- | --- |
| PassengerId | 0.00 | 0.00 |
| HomePlanet | 201.00 | 2.31 |
| CryoSleep | 217.00 | 2.50 |
| Cabin | 199.00 | 2.29 |
| Destination | 182.00 | 2.09 |
| Age | 179.00 | 2.06 |
| VIP | 203.00 | 2.34 |
| RoomService | 181.00 | 2.08 |
| FoodCourt | 183.00 | 2.11 |
| ShoppingMall | 208.00 | 2.39 |
| Spa | 183.00 | 2.11 |
| VRDeck | 188.00 | 2.16 |
| Name | 200.00 | 2.30 |
| Transported | 0.00 | 0.00 |

### Test

| Column | missing_count | missing_percent |
| --- | --- | --- |
| PassengerId | 0.00 | 0.00 |
| HomePlanet | 87.00 | 2.03 |
| CryoSleep | 93.00 | 2.17 |
| Cabin | 100.00 | 2.34 |
| Destination | 92.00 | 2.15 |
| Age | 91.00 | 2.13 |
| VIP | 93.00 | 2.17 |
| RoomService | 82.00 | 1.92 |
| FoodCourt | 106.00 | 2.48 |
| ShoppingMall | 98.00 | 2.29 |
| Spa | 101.00 | 2.36 |
| VRDeck | 80.00 | 1.87 |
| Name | 94.00 | 2.20 |

## 5. Duplicate Analysis

| Dataset | Exact duplicate rows | Duplicate `PassengerId` values |
| --- | ---: | ---: |
| train | 0 | 0 |
| test | 0 | 0 |
| sample submission | 0 | 0 |

## 6. Target Distribution

`Transported` exists in train, has dtype `bool`, and has 0 missing values.

| Transported | Count | Percent |
| --- | --- | --- |
| True | 4378.00 | 50.36 |
| False | 4315.00 | 49.64 |

## 7. Numerical Feature Audit

| Column | count | mean | std | min | 25% | median | 75% | max | zero_count | negative_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Age | 8514.00 | 28.83 | 14.49 | 0.00 | 19.00 | 27.00 | 38.00 | 79.00 | 178.00 | 0.00 |
| RoomService | 8512.00 | 224.69 | 666.72 | 0.00 | 0.00 | 0.00 | 47.00 | 14327.00 | 5577.00 | 0.00 |
| FoodCourt | 8510.00 | 458.08 | 1611.49 | 0.00 | 0.00 | 0.00 | 76.00 | 29813.00 | 5456.00 | 0.00 |
| ShoppingMall | 8485.00 | 173.73 | 604.70 | 0.00 | 0.00 | 0.00 | 27.00 | 23492.00 | 5587.00 | 0.00 |
| Spa | 8510.00 | 311.14 | 1136.71 | 0.00 | 0.00 | 0.00 | 59.00 | 22408.00 | 5324.00 | 0.00 |
| VRDeck | 8505.00 | 304.85 | 1145.72 | 0.00 | 0.00 | 0.00 | 46.00 | 24133.00 | 5495.00 | 0.00 |

All six audited numeric columns have zero negative values. Zero spending is retained as a meaningful observed value. The table reports extremes for later modelling review; this audit does not remove outliers.

## 8. Categorical Feature Audit

The boolean-like columns are stored as `object` because of missing values. Their observed non-missing values are consistently `True` and `False`.

| Dataset | Column | Value | Count |
| --- | --- | --- | --- |
| test | CryoSleep | False | 2640.00 |
| test | CryoSleep | True | 1544.00 |
| test | CryoSleep |  | 93.00 |
| test | VIP | False | 4110.00 |
| test | VIP | True | 74.00 |
| test | VIP |  | 93.00 |
| train | CryoSleep | False | 5439.00 |
| train | CryoSleep | True | 3037.00 |
| train | CryoSleep |  | 217.00 |
| train | VIP | False | 8291.00 |
| train | VIP | True | 199.00 |
| train | VIP |  | 203.00 |

`HomePlanet`, `Destination`, `CryoSleep`, and `VIP` have no empty strings, whitespace-only values, surrounding whitespace, or textual null tokens in either raw dataset. `PassengerId`, `Cabin`, and `Name` are high-cardinality fields; their values are preserved unchanged.

## 9. PassengerId Investigation

All 8,693 non-missing training `PassengerId` values match `dddd_dd` (four digits, underscore, two digits); no training identifiers are duplicated. The same pattern check is performed in the reusable audit code for test. The first component appears to encode a travelling group and the second a passenger position within that group, consistent with the dataset description. No group-derived feature is created in this phase.

## 10. Cabin Investigation

`Cabin` has 199 missing training values. All 8,494 non-missing training cabin values match the observed `deck/number/side` pattern, and the observed component-count distribution is {3: 8494}. No deck, cabin-number, or side feature is extracted yet.

## 11. Data Quality Issues

- No exact duplicate rows were found in train or test.
- Missing values occur in multiple passenger attributes; they are preserved for later preprocessing.
- `PassengerId` and `Name` are high-cardinality fields and require deliberate treatment in a later feature-engineering phase.
- `Cabin` has missing records but every non-missing training value matches the observed three-part `deck/number/side` pattern.

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
