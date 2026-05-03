# Customer Churn Prediction System

A beginner-friendly mini project for Foundations of Data Science. The system stores customer details, predicts customer churn using a simple Logistic Regression model, and displays attractive charts in a pink-purple theme.

## Features

- Home page with intuitive navigation
- Customer details page with table, search, filter, add, and delete functions
- Churn prediction page with probability, risk level, and suggestion
- Dashboard with summary metrics and colorful Chart.js visualizations
- Sample dataset with 100+ customer rows in `data/customers.csv`

## Technologies

- Python
- Flask
- HTML / CSS
- JavaScript
- Pandas
- Scikit-learn
- Chart.js
- CSV dataset

## Run the Project

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
python app.py
```

3. Open the browser and go to:

```text
http://127.0.0.1:5000
```

## Notes

- The dataset is loaded from `data/customers.csv`.
- The Logistic Regression model trains each time the server starts.
- The design uses a pink and purple gradient theme for a modern demo presentation.
