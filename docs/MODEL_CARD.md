# Spaceship Titanic Model Card

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

On the validation split: accuracy 0.8016, precision 0.7940, recall 0.8185, F1 0.8061, ROC-AUC 0.9039. Confusion matrix: TN=677, FP=186, FN=159, TP=717.

## Error Analysis

Validation error rates were summarized by age group and CryoSleep. They are descriptive subgroup summaries and should not be interpreted as causal.

## Limitations

Metrics are tied to this project split and do not guarantee production performance. Missingness, subgroup sample size, train/test distribution changes, and competition-specific data limits remain relevant.

## Reproducibility

Use Python 3.12 with project dependencies and load `C:/Users/davro/OneDrive/Desktop/ml_projects/spaceship-titanic-ml/model/spaceship_titanic_pipeline.joblib` with joblib. The metadata file records feature configuration and split settings.

## Deployment Notes

The artifact requires the same input column schema and compatible package versions. No API or browser deployment is created in this phase.
