# Loan Prediction Model Summary

## Notebook Summary

This notebook focuses on building and evaluating machine learning models to predict loan approval status from the `loan_approval_dataset.csv` dataset.

### Key steps

1. **Data Loading and Initial Exploration**
   - Loaded the dataset into a Pandas DataFrame.
   - Inspected shape, preview rows, null values, data types, and summary statistics.
2. **Data Cleaning and Feature Engineering**
   - Dropped `loan_id` as it is not predictive.
   - Created `Total_asset` by summing `bank_asset_value`, `residential_assets_value`, `luxury_assets_value`, and `commercial_assets_value`.
   - Removed the original asset columns after deriving `Total_asset`.
   - Cleaned whitespace from column names and string values.
   - Encoded categorical features such as `education`, `self_employed`, and `loan_status`.
3. **Data Splitting and Preprocessing**
   - Split data into training and test sets with 20% test size.
   - Applied `StandardScaler` to `income_annum` and `loan_amount`.
   - Applied `PowerTransformer` to `Total_asset` to reduce skew.
   - Combined preprocessing with a `ColumnTransformer`.
4. **Model Training and Evaluation**
   - Trained Logistic Regression on scaled features.
   - Trained Random Forest and Decision Tree classifiers on original features.
   - Evaluated using accuracy, confusion matrices, classification reports, and cross-validation.

## Results

### Logistic Regression

- Test accuracy: approximately **90.6%**
- Confusion matrix:

```
[[276  42]
 [ 38 498]]
```

- Classification metrics were generally between 0.87 and 0.93.

### Random Forest Classifier

- Test accuracy: approximately **97.7%**
- 5-fold cross-validation mean: approximately **97.8%**
- Confusion matrix:

```
[[305  13]
 [  7 529]]
```

- Feature importances:
  - `cibil_score`: 0.839
  - `loan_term`: 0.061
  - `loan_amount`: 0.034

### Decision Tree Classifier

- Test accuracy: approximately **97.5%**
- 5-fold cross-validation mean: approximately **97.9%**

## Final Model

The Random Forest model was selected as the final production model due to its high accuracy and stable cross-validation performance.

- Saved model file: `loan_model.pkl`

## Notes

- The notebook shows that `cibil_score` is the strongest predictor.
- `Total_asset` is a derived feature used to summarize multiple asset columns.
- The preprocessing pipeline must match inference preprocessing for correct predictions.

## Files

- `LoanPredictionModel.ipynb` — notebook with the full modeling workflow
- `loan_approval_dataset.csv` — dataset used for training
- `loan_model.pkl` — saved Random Forest model
- `app.py` — Streamlit app for making predictions using `loan_model.pkl`
