# Spaceship Titanic - Model Results

## 1. Experiment Setup

Experiments use `data/processed/train_clean.csv`, target `Transported`, and a stratified 80/20 split with `random_state=42` (6,954 train / 1,739 validation). Every model uses the same target-free feature engineer followed by median-impute/scale numerical preprocessing and most-frequent-impute/one-hot categorical preprocessing. Kaggle test data is not used for model selection or metrics.

## 2. Models Tested

Logistic Regression (`max_iter=2000`), Decision Tree, Random Forest (200 estimators), and Gradient Boosting; all use `random_state=42`.

## 3. Validation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7901 | 0.7880 | 0.7979 | 0.7930 | 0.8835 |
| Decision Tree | 0.7591 | 0.7654 | 0.7523 | 0.7588 | 0.7595 |
| Random Forest | 0.8120 | 0.8385 | 0.7763 | 0.8062 | 0.8952 |
| Gradient Boosting | 0.8016 | 0.7940 | 0.8185 | 0.8061 | 0.9039 |

## 4. Cross-Validation Results

Five-fold stratified cross-validation (`shuffle=True`, `random_state=42`) was run on the training-fold records. Values are mean and standard deviation across folds.

| Model | Accuracy mean | Accuracy std | F1 mean | F1 std | ROC-AUC mean | ROC-AUC std |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7893 | 0.0035 | 0.7953 | 0.0043 | 0.8779 | 0.0015 |
| Decision Tree | 0.7409 | 0.0135 | 0.7424 | 0.0127 | 0.7410 | 0.0135 |
| Random Forest | 0.8001 | 0.0057 | 0.7940 | 0.0059 | 0.8811 | 0.0042 |
| Gradient Boosting | 0.8064 | 0.0070 | 0.8122 | 0.0067 | 0.8933 | 0.0029 |

## 5. Confusion Matrices

- **Logistic Regression:** TN=675, FP=188, FN=177, TP=699
- **Decision Tree:** TN=661, FP=202, FN=217, TP=659
- **Random Forest:** TN=732, FP=131, FN=196, TP=680
- **Gradient Boosting:** TN=677, FP=186, FN=159, TP=717

## 6. ROC-AUC

Validation ROC-AUC measurements are in the validation table and all four curves are shown in `roc_curve_comparison.png`.

## 7. Feature Importance

`feature_importance.png` shows Random Forest internal impurity-based feature importance after preprocessing. It describes that fitted model's split criterion usage; it is not causal importance.

## 8. Observations

The tables report holdout and cross-validation measurements for fixed baselines. Differences should be interpreted alongside fold variation rather than as a final selection decision.

## 9. Model Selection Considerations

Step 05 should consider validation results, cross-validation stability, interpretability, inference complexity, and model size before selecting a final artifact. No final model is selected here.


## Final Evaluation

The selected Gradient Boosting configuration was re-fit on the original 6,954-row training fold and evaluated on the unchanged 1,739-row validation fold. At the default 0.50 probability threshold: accuracy=0.8016, precision=0.7940, recall=0.8185, F1=0.8061, ROC-AUC=0.9039. ROC-AUC measures ranking discrimination over thresholds; this single holdout remains only one sample of future performance.

Its validation confusion matrix is TN=677, FP=186, FN=159, TP=717; `final_confusion_matrix.png` records the same values. `precision_recall_comparison.png` and `cv_stability.png` summarize the fixed Step 04 comparisons.

## Error Analysis

Error rates are descriptive validation-set summaries, not causes of error. Age-group results:

| AgeGroup | records | errors | error_rate |
| --- | --- | --- | --- |
| 0-12 | 141.0000 | 45.0000 | 0.3191 |
| 45-64 | 237.0000 | 50.0000 | 0.2110 |
| 18-29 | 660.0000 | 135.0000 | 0.2045 |
| 13-17 | 155.0000 | 30.0000 | 0.1935 |
| 30-44 | 480.0000 | 78.0000 | 0.1625 |
| 65+ | 26.0000 | 3.0000 | 0.1154 |
| Missing | 40.0000 | 4.0000 | 0.1000 |

CryoSleep results:

| CryoSleep | records | errors | error_rate |
| --- | --- | --- | --- |
| Missing | 49.0000 | 11.0000 | 0.2245 |
| False | 1089.0000 | 224.0000 | 0.2057 |
| True | 601.0000 | 110.0000 | 0.1830 |

The dedicated charts show these subgroup rates. Small groups should be interpreted cautiously.

## Production Candidate

Gradient Boosting is the production candidate configuration. Its fixed holdout ROC-AUC (0.9039) and five-fold mean ROC-AUC (0.8933 ± 0.0029) are the highest recorded; its five-fold mean F1 is also the highest recorded (0.8122 ± 0.0067). This is an engineering choice from the measured baselines, not a guarantee of production performance or a causal conclusion.

## Final Artifact

The final pipeline, including feature engineering, preprocessing, and GradientBoostingClassifier, was re-fit on all 8,693 labelled processed rows and saved to `C:/Users/davro/OneDrive/Desktop/ml_projects/spaceship-titanic-ml/model/spaceship_titanic_pipeline.joblib`. It accepts raw passenger feature columns (without `Transported`) and supports `predict` and `predict_proba`.
