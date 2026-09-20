# Project Plan

## Objective

Build a portfolio-quality end-to-end machine learning case study for Kaggle's Spaceship Titanic competition. The central task is to predict the binary `Transported` outcome from recovered passenger records while documenting a reproducible data-science workflow and presenting it through a static, cinematic editorial website.

## Competition Context

Spaceship Titanic asks participants to predict whether a passenger was transported to another dimension after the ship's collision with a spacetime anomaly. The supplied training data contains the `Transported` target; the test data does not.

## Machine Learning Workflow

1. Audit the raw Kaggle files without modifying them.
2. Clean data reproducibly and save derived outputs to `data/processed/`.
3. Perform exploratory data analysis and record evidence-based findings.
4. Engineer and validate candidate features.
5. Build preprocessing pipelines that avoid train/test leakage.
6. Train and compare baseline and candidate classifiers.
7. Evaluate with appropriate holdout and cross-validation evidence.
8. Select and document a final model.
9. Decide how to reproduce the selected model and preprocessing in the browser.

## Website Workflow

The static site will follow this narrative: **The Incident → The Voyage → The Anomaly → The Evidence → Data Cleaning → Exploration → Patterns → Feature Engineering → Model Training → Model Evaluation → The Predictor → Passenger Scanner → Final Transmission**.

It will use HTML, CSS, and Vanilla JavaScript only. The visual direction is cinematic, editorial, scientific, technical, restrained, and typography-led. Custom and generated illustrations will be preserved as project assets.

## Deployment Strategy

The website will be deployed as a static site on Netlify. No API, backend, or server folder is part of this project. Browser prediction will be designed only after the final model and preprocessing requirements are understood.

## Project Phases

## Status

- STEP 01 - DATA AUDIT & CLEANING: COMPLETED
- STEP 02 - FULL EDA: COMPLETED
- STEP 03 - FEATURE ENGINEERING & PREPROCESSING: COMPLETED
- STEP 04 - MODEL TRAINING: COMPLETED
- STEP 05 - MODEL EVALUATION: COMPLETED

1. Foundation and documentation — complete.
2. Data audit and cleaning. — completed.
3. Exploratory data analysis. — completed.
4. Feature engineering and preprocessing.
5. Model training and comparison.
6. Evaluation, selection, and model explanation.
7. Browser-compatible inference design.
8. Cinematic website implementation.
9. Documentation, static deployment, and final review.

## Future Tasks

- Add the official Kaggle data to `data/raw/` without altering it.
- Create data-audit and cleaning functions in `src/data/`.
- Populate the EDA and model-result documents only with measured findings.
- Define a reproducible experiment protocol.
- Select a browser inference strategy after final-model evaluation.
- Build and test the responsive Netlify-ready website.
