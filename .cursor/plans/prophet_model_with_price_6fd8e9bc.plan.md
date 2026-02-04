---
name: Prophet Model with Price
overview: Implement a Prophet model for each series (143 total) using price as a regressor, with log transformation and proper normalization for robust predictions.
todos:
  - id: train-function
    content: Create train_prophet_for_series function with log transform, logistic growth, and price regressor
    status: completed
  - id: training-loop
    content: Implement training loop for all 143 series with progress tracking
    status: completed
  - id: predictions
    content: Generate predictions on test data and calculate MAE, sMAPE, WAPE metrics
    status: completed
isProject: false
---

# Prophet Model Implementation with Price Regressor

## Context

The notebook [5_Prophet.ipynb](final_version_TFG/5_Prophet.ipynb) has data prepared with:

- 143 unique series (brand + supermarket + variant + pack_size combinations)
- Train/test split: train until 2023-06-30, test from 2023-07-01 to 2023-12-31
- Variables available: `date`, `volume.sales`, `price`, and categorical variables

## Implementation for Cell 6

The code will include:

### 1. Function to Train Prophet for a Single Series

```python
def train_prophet_for_series(series_data_train, use_price_regressor=True, seasonality_mode='multiplicative'):
```

Key features:

- **Log transformation**: Using `log1p(volume.sales)` to stabilize variance and handle near-zero values
- **Logistic growth**: With `floor=0` and `cap` based on max historical value (ensures non-negative predictions)
- **Price regressor**: Normalized price `(price - mean) / std` only when price varies in the series
- **Seasonality**: Multiplicative yearly seasonality (appropriate for sales data)
- **Changepoint detection**: Conservative `changepoint_prior_scale=0.05`

### 2. Training Loop for All Series

Iterates over all 143 series and stores:

- `prophet_models`: Dictionary with trained models
- `prophet_models_info`: Dictionary with metadata (cap, price normalization params, whether price was used)

### 3. Prediction and Metrics Calculation

After training, generates predictions for the test period and calculates:

- MAE (Mean Absolute Error)
- sMAPE (Symmetric Mean Absolute Percentage Error)
- WAPE (Weighted Absolute Percentage Error)

## Key Data Format Conversion

Prophet requires specific column names:

- `ds`: date column
- `y`: target variable (log-transformed `volume.sales`)
- `floor`, `cap`: bounds for logistic growth
- `price_normalized`: normalized price regressor

## Code Structure

```python
€# Cell 6 content:

# 1. Define training function
def train_prophet_for_series(...):
    # Prepare data for Prophet (ds, y, floor, cap)
    # Add normalized price if it varies
    # Create and fit Prophet model
    # Return model and metadata

# 2. Train all series
prophet_models = {}
prophet_models_info = {}

for series_id in train_data['series_id'].unique():
    # Train model
    # Store model and info

# 3. Generate predictions on test set
# 4. Calculate and display metrics
```

## Notes

- Series where price is constant will be trained without the price regressor (only trend + seasonality)
- Predictions will be reverted from log scale using `expm1()` before calculating metrics
- Error handling included for series that fail to train
