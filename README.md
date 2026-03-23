# Customer Churn Predictor

A simple machine learning project for customer churn prediction using Python.

## Project Overview

This project demonstrates building, training, and evaluating a model to predict whether a customer will churn based on input features.

## Files

- `customer_churn_system1.py` - Main script that loads data, trains a model, and evaluates predictions.
- `churn setup.ipynb` - Jupyter notebook for exploratory data analysis and modeling workflow.
- `requirements.txt` - Python dependencies for this project.
- `README.md` - This documentation file.

## Setup Instructions

1. Create and activate Python environment (recommended):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the script:

```powershell
python customer_churn_system1.py
```

4. Open the notebook for interactive exploration:

```powershell
jupyter notebook "churn setup.ipynb"
```

## Expected behavior

- Model training should run and print metrics like accuracy and confusion matrix.
- Prediction output indicates whether individual customers are expected to churn.

## How to contribute

- Add or clean data inputs in the notebook.
- Improve model selection, featurization, preprocessing steps.
- Add tests for data validation and prediction quality.

## License

Add your license details here as needed.
