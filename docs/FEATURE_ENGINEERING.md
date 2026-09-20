# Feature Engineering and Preprocessing

## Original Features

The model-input candidates are `HomePlanet`, `CryoSleep`, `Destination`, `Age`, `VIP`, and the five spending columns. These are retained because they are low-cardinality or numeric passenger attributes with EDA-supported descriptive relationships to the target.

## Engineered Features

| Feature | Source | Logic | Reason | Missing-value behavior |
| --- | --- | --- | --- | --- |
| `CabinDeck` | `Cabin` | First component of `deck/number/side` | Deck rates differed descriptively in EDA. | Missing cabin produces missing category, then most-frequent imputation. |
| `CabinNumber` | `Cabin` | Numeric second component; malformed values coerce to missing | Preserves structured location information for later evaluation. | Median imputation inside the fitted numerical pipeline. |
| `CabinSide` | `Cabin` | Third component | S/P side showed descriptive rate differences. | Missing category, then most-frequent imputation. |
| `GroupId` | `PassengerId` | Prefix before `_` | Intermediate only; used to calculate GroupSize. | Not supplied to the model. |
| `GroupSize` | `PassengerId` | Count of `GroupId` values in the training fold during `fit` | Captures verified travelling-group structure without target aggregation. | Unseen validation/test groups receive 1. |
| `TotalSpending` | Five spending columns | Row sum with `min_count=1` | Captures overall onboard spending; EDA found zero inflation and skew. | Remains missing only if all five values are missing, then median-imputed. |
| `HasSpending` | `TotalSpending` | `TotalSpending > 0` | Separates legitimate zero spending from positive spending. | Missing only if TotalSpending is missing; categorical imputation applies. |

All engineering is target-free. `Transported` is rejected by the feature-engineering transformer.

## Removed or Excluded Features

- `PassengerId` is unique for all 8,693 training rows. Its original value is excluded after deriving the verified group prefix and train-fit `GroupSize`.
- `Name` has 8,473 unique non-missing values and is excluded from the baseline-ready matrix: it is high-cardinality text with no deliberate text feature design in this phase.
- `Cabin` has 6,560 unique non-missing values and is excluded after parsing deck, numeric location, and side.
- `GroupId` is an intermediate high-cardinality identifier, excluded after its safe count feature is derived.

## Final Feature Groups

### Numerical

`Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`, `CabinNumber`, `GroupSize`, `TotalSpending`

### Categorical

`HomePlanet`, `CryoSleep`, `Destination`, `VIP`, `CabinDeck`, `CabinSide`, `HasSpending`

### Target

`Transported`

## Preprocessing Architecture

`TitanicFeatureEngineer` runs first. A `ColumnTransformer` then applies median imputation and `StandardScaler` to numerical features, and most-frequent imputation plus `OneHotEncoder(handle_unknown="ignore")` to categorical features. The full graph is a scikit-learn `Pipeline`; no manual dummy variables or full-dataset fitted statistics are saved.

## Leakage Audit

- `Transported` is dropped before the split inputs and the transformer raises if it is supplied.
- The split is stratified (`test_size=0.20`, `random_state=42`) and uses only `train_clean.csv`.
- Engineering group counts, imputation values, scaling statistics, and one-hot categories are fit only on training-fold features.
- Validation and test data are transformed only after fitting; they are never passed to `fit`.
- No target-derived feature, group transportation rate, or target aggregation is implemented.
- An unseen categorical-value probe is transformed successfully with `handle_unknown="ignore"`.

## Validation Status

The implementation specifies a 6,954-row training fold and a 1,739-row validation fold. Its smoke test is designed to verify consistent transformed dimensions, no non-finite transformed values, successful test transformation, and safe handling of an unseen categorical value. It could not run in the current environment because Windows Application Control blocks scikit-learn's compiled `_radius_neighbors` DLL under the available Python 3.14 runtime. No classifier was fit. Run `python -m src.features.run_preprocessing` in an environment with a permitted, compatible scikit-learn installation before marking Step 03 complete.
