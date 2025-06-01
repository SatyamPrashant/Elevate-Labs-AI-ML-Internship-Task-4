# Elevate Labs AI/ML Internship Task 4: Classification with Logistic Regression

## Repository Description (2 lines)
A complete end‐to‐end pipeline that builds and evaluates a Logistic Regression model to classify breast tumors as benign or malignant, including feature scaling, model evaluation (confusion matrix, precision, recall, ROC‐AUC), threshold tuning, and coefficient interpretation.  
Includes a saved model pipeline (`logistic_regression_model.pkl`) and detailed Jupyter notebook for reproducibility.

---

## Dataset
We use the **Breast Cancer Wisconsin (Diagnostic) dataset** available via scikit-learn.  
- 569 samples, 30 numeric features, target: 0 = benign, 1 = malignant.

## Libraries / Tools
- Python 3.x  
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn (LogisticRegression, metrics)  

---

## Steps Performed

1. **Imports & Data Loading**  
   - Loaded dataset using `sklearn.datasets.load_breast_cancer(as_frame=True)`.  
   - Renamed target column to `diagnosis` (0/1).

2. **Initial Inspection**  
   - Checked dataset shape, missing values, and class balance.  
   - Reviewed summary statistics of all features.

3. **Train/Test Split**  
   - Separated features (`X`) and target (`y`).  
   - Performed an 80/20 split with `stratify=y` to maintain class proportions.

4. **Feature Scaling**  
   - Standardized all numeric features to mean ≈ 0, std ≈ 1 using `StandardScaler`.  
   - Ensured test set used the same scaling parameters.

5. **Fit Logistic Regression**  
   - Trained `LogisticRegression(solver='liblinear')` on scaled training data.  
   - Predicted on the test set.

6. **Model Evaluation**  
   - Computed confusion matrix, classification report (precision, recall, F1).  
   - Calculated accuracy, precision, recall, and F1‐score.  

7. **ROC Curve & AUC**  
   - Plotted ROC curve from predicted probabilities.  
   - Computed AUC.

8. **Threshold Tuning**  
   - Adjusted classification threshold to achieve ≥95% recall.  
   - Observed corresponding precision at the new threshold.

9. **Sigmoid Visualization**  
   - Plotted the sigmoid function to understand the logistic transformation.

10. **Class Imbalance Handling (Optional)**  
    - Showed class distribution in original data (~37% malignant).  
    - Re‐fitted with `class_weight='balanced'` to boost minority class.  
    - Re‐evaluated metrics.

11. **Interpretation of Coefficients & Odds Ratios**  
    - Extracted feature coefficients from the trained model.  
    - Calculated odds ratios (`exp(coef)`) to see feature impact on malignant odds.

12. **Save Final Model**  
    - Pickled the trained classifier as `logistic_regression_model.pkl`.

---

## How to Reproduce

1. Clone this repo.  
2. Open `task4_logistic_regression.ipynb` in Jupyter or JupyterLab.  
3. Run each cell in order.  
4. (Optional) Load the saved model in another script:
   ```python
   import joblib
   clf = joblib.load("logistic_regression_model.pkl")
   # Prepare new samples, scale them with the same StandardScaler, and call clf.predict()
