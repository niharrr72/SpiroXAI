# SpiroXAI

SpiroXAI is a clinical diagnostic system for lung disease, utilizing advanced machine learning models (DNN, XGBoost, LIghtGBM, FT-Transformer).

## Features
- **API Backend**: Built with FastAPI to serve predictions and explainability (SHAP).
- **Machine Learning**: Predicts lung diseases based on patient features (spirometry and demographic data) with confidence intervals.
- **Reporting**: Automatically generates PDF reports containing patient metrics, models prediction, and features' impacts.
- **Explainable AI**: Features integrated SHAP support for medical transparency.

## Project Structure
- FastAPI application code for endpoints, authentication, user management, predictions, and report generation.
- `saved_models/`: Pre-trained models (DNN, XGBoost, LightGBM, FT-Transformer) and encoders/scalers used for inference.
- Notebooks: Development notebooks for EDA and model training (`h200.ipynb`, `new-python-dummy h100.ipynb`, etc.)
- Explainability Plots: SHAP summary plots and various evaluation visualizations (confusion matrices, ROC curves, feature importance).


## Requirements
To install dependencies, please run:
```bash
pip install -r backend/requirements.txt
```

## Running the Backend
From the root directory or `backend` directory, run the FastAPI application:
```bash
cd backend
python main.py
```
The API is typically exposed on `http://0.0.0.0:8000`. 
