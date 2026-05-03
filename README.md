# Weather Temperature Forecasting — WiDS 2025

Kaggle WiDS Datathon 2025 competition project.
Predicts 14-day average temperature (tmp2m) using CatBoost with time-series cross-validation.

## Results

| Metric | Baseline (unoptimised) | This model |
|--------|----------------------|------------|
| MAE    | ~4.5°C               | ~1.0°C     |
| CV     | 5-fold TimeSeriesSplit | 5-fold TimeSeriesSplit |

## My Role
Model architecture design, hyperparameter tuning, and training pipeline.
Team project — teammates handled data collection and feature engineering.

## Project Structure

├── main/          # Entry point
├── model/         # CatBoost model class
├── notebook/      # EDA, preprocessing, modeling, evaluation
└── README.md

## Key Techniques
- CatBoost Regressor with 2000 iterations, depth=8, learning_rate=0.025
- 25 selected meteorological features (pressure, wind, sea surface temp, etc.)
- 5-fold TimeSeriesSplit cross-validation

## Dataset
Kaggle WiDS Datathon 2025: https://www.kaggle.com/competitions/widsdatathon2025

## Requirements
pip install -r requirements.txt
