# Housing Starts Forecasting: TabPFN vs XGBoost

A Jupyter notebook demonstrating TabPFN for time series regression compared to XGBoost using U.S. housing starts data from FRED.

## Overview

This project explores the application of **TabPFN** (Tabular Prior-Data Fitted Network) for time series forecasting in a Jupyter notebook environment. We compare TabPFN's zero-shot learning capabilities against traditional XGBoost for predicting housing starts with built-in uncertainty quantification.

<img width="688" height="390" alt="image" src="https://github.com/user-attachments/assets/1d390036-a092-49cc-a33a-5fc6ef73734a" />


*TabPFN provides uncertainty intervals while XGBoost gives point forecasts only*

## Key Features

- 📊 **Interactive Analysis**: Step-by-step Jupyter notebook walkthrough
- 🚀 **Zero Hyperparameter Tuning**: TabPFN works out-of-the-box
- 📈 **Uncertainty Quantification**: TabPFN provides prediction intervals
- ⚡ **Fast Predictions**: Instant forecasting after training
- 🔄 **Side-by-side Comparison**: TabPFN vs XGBoost analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/maghwa/housing-forecasting-tabpfn.git
cd housing-forecasting-tabpfn

# Launch Jupyter notebook
jupyter notebook housing_forecast.ipynb
```

## Dependencies

```bash
pip install tabpfn==2.0.9
pip install xgboost==3.0.2
pip install pandas==2.2.3
pip install numpy==2.2.6
pip install scikit-learn==1.6.1
pip install requests==2.32.3
pip install matplotlib==3.10.3
pip install jupyter
```

## Notebook Structure

### 📚 **housing_forecast.ipynb**

The notebook is organized into the following sections:

1. **Setup & Imports**
   - Library installations
   - Configuration settings
   - Environment setup

2. **Data Import & Preprocessing**
   - Download housing starts data from FRED API
   - Data cleaning and preparation
   - Feature engineering (month, time trend)

3. **Exploratory Data Analysis**
   - Time series visualization
   - Trend and seasonality analysis

4. **Model Implementation**
   - TabPFN configuration and training
   - XGBoost baseline implementation
   - Prediction generation

5. **Results & Comparison**
   - Performance metrics (MAE)
   - Visualization of forecasts
   - Uncertainty interval analysis

6. **Discussion & Insights**
   - Model strengths and limitations
   - Use case recommendations

## Quick Start

1. **Open the notebook:**
   ```bash
   jupyter notebook housing_forecast.ipynb
   ```

2. **Run all cells** or execute step-by-step to see:
   - Real-time data download from FRED
   - Interactive model training
   - Dynamic forecast visualization

3. **Experiment with parameters:**
   - Change forecast horizon
   - Modify feature engineering
   - Adjust prediction intervals

## Notebook Highlights

### 🔧 **Configuration**
```python
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"  # Lift 1k-row limit
```

### 📊 **Data Pipeline**
```python
# Direct FRED API integration
CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=HOUSTNSA"
df = pd.read_csv(io.StringIO(requests.get(CSV_URL).text))
```

### 🤖 **Zero-Shot Learning**
```python
tab = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
tab.fit(X_train, y_train)  # No hyperparameter tuning needed!
```

### 📈 **Uncertainty Quantification**
```python
y_median = tab.predict(X_test, output_type="median")
q10, q90 = tab.predict(X_test, output_type="quantiles", quantiles=[0.1, 0.9])
```

## Results Summary

| Metric | TabPFN | XGBoost |
|--------|---------|---------|
| **Training Time** | ~1 second | Variable |
| **Hyperparameter Tuning** | None | Required |
| **Uncertainty** | ✅ Quantiles | ❌ Point only |
| **MAE** | Competitive | Baseline |
