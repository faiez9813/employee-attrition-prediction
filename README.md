# Employee Attrition Prediction

A machine learning project that predicts employee attrition using a Random Forest Classifier, with an emphasis on handling severe class imbalance, threshold tuning, and translating model predictions into a usable function for new inputs.

## Overview

This project applies binary classification to predict whether an employee is likely to leave the company, using the IBM HR Analytics Employee Attrition dataset.

## Dataset

- Rows: 1,470 employees
- Features: 31 (after removing zero-variance columns)
- Target: Attrition (1 = Left, 0 = Stayed)
- Class balance: 83.9% Stayed, 16.1% Left (significant imbalance, small minority class size)

## Workflow

1. Data Cleaning
   - Identified and removed three zero-variance columns (EmployeeCount, Over18, StandardHours) that carried no predictive information
   - Removed EmployeeNumber, a unique identifier with no predictive value

2. Exploratory Data Analysis
   - Employees working overtime had a 30.5% attrition rate versus 10.4% for those who did not, roughly three times higher
   - BusinessTravel frequency showed a clear, gradual relationship with attrition: 8% (Non-Travel), 15% (Travel_Rarely), 25% (Travel_Frequently), supporting its treatment as an ordinal feature
   - MonthlyIncome showed some high values on a boxplot, but these were judged to reflect genuine senior-level salaries rather than data errors, and were left unchanged

3. Feature Encoding
   - Binary columns (Gender, OverTime) mapped to 0/1
   - BusinessTravel encoded as ordinal, based on the natural order observed in the EDA
   - Remaining nominal columns (Department, EducationField, JobRole, MaritalStatus) encoded using One-Hot Encoding

4. Modeling
   - Trained a Random Forest Classifier (100 trees, max depth 6) with class_weight="balanced"
   - Used a stratified train-test split to preserve the class ratio

5. Threshold Tuning
   - Tested thresholds from 0.2 to 0.5
   - Threshold 0.5 gave the highest overall accuracy (0.84) but a low recall (0.36) for the minority class
   - Threshold 0.4 was selected as the best balance, improving recall to 0.62 with an acceptable precision trade-off (0.39), since in this context, failing to identify an at-risk employee is more costly than flagging a low-risk one for additional attention

6. Prediction Function
   - Built a wrapper function that accepts plain, human-readable inputs (e.g., "Male", "Travel_Frequently") and internally handles all encoding and column alignment before calling the trained model, mirroring how a production backend would handle form input

## Results

| Metric (Attrition = Yes) | Threshold 0.5 | Threshold 0.4 |
|---|---|---|
| Precision | 0.49 | 0.39 |
| Recall | 0.36 | 0.62 |
| F1-Score | 0.41 | 0.48 |

ROC-AUC Score: 0.766

## Key Takeaways

- Not every dataset yields excellent performance; a small minority class size (237 cases) and the inherent complexity of predicting human behavior limited what the model could learn, and this is reported as a genuine finding rather than adjusted to look better
- A monotonic, gradual relationship between a feature and the target (like BusinessTravel) is a stronger signal of a genuine pattern than an extreme, near-deterministic one
- Wrapping a trained model in a function that handles preprocessing internally is a practical step toward making a model usable beyond the notebook it was built in

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (RandomForestClassifier, train_test_split, classification_report, roc_auc_score)
- Matplotlib, Seaborn

## Files

employee_attrition_prediction.ipynb: full analysis and modeling notebook

## How to Run

1. Clone this repository
2. Open employee_attrition_prediction.ipynb in Jupyter Notebook or Google Colab
3. Download the dataset from Kaggle (IBM HR Analytics Employee Attrition Dataset) and place the CSV file in the same directory
4. Run all cells
