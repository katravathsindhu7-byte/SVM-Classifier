# SVM-Classifier
# SVM Classifier - California Housing

## Overview

This project implements a Support Vector Machine (SVM) classifier using the California Housing dataset. It classifies house values into Low, Medium, and High categories.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Google Colab

## Dataset

The California Housing dataset is loaded using Scikit-learn.

The target variable `MedHouseVal` is measured in units of $100,000.

House value categories:

* Low: below $150,000
* Medium: $150,000 to below $300,000
* High: $300,000 or above

## Algorithm

* Algorithm: Support Vector Machine (SVM)
* Kernel: RBF
* Regularization parameter: C = 1.0
* Feature scaling: StandardScaler
* Training/testing split: 80% / 20%, with up to 6,000 training samples

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1-score
* Confusion Matrix

The actual scores are saved in `svm_classifier_metrics.csv`.

## Repository Files

* `SVM_Classifier_California_Housing.ipynb` — notebook containing code and execution outputs
* `svm_classifier_predictions.csv` — actual and predicted categories
* `svm_classifier_metrics.csv` — evaluation metrics

## How to Run

1. Open the notebook in Google Colab.
2. Run all cells in order.
3. Review the model evaluation results.
4. Download the generated CSV files if required.

## Author

K. Sindhu

# ============================================================
# PROJECT: California Housing Price Prediction using SVR
# Algorithms: SVR with Linear Kernel and RBF Kernel
# Platform: Google Colab
# ============================================================

# STEP 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# STEP 2: Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())

# Save the original dataset
df.to_csv("california_housing.csv", index=False)

# STEP 3: Separate features and target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# STEP 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# STEP 5: Define two SVR models
# Scaling is fitted only on training data through each pipeline.
models = {
    "SVR Linear Kernel": Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="linear", C=1.0, epsilon=0.1))
    ]),

    "SVR RBF Kernel": Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10.0,
                    epsilon=0.1, gamma="scale"))
    ])
}

# STEP 6: Train models, predict and calculate metrics
results = []
predictions = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    trained_models[name] = model
    predictions[name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

# STEP 7: Display model comparison
metrics_df = pd.DataFrame(results)

print("\n========== SVR MODEL COMPARISON ==========")
print(metrics_df.round(4).to_string(index=False))

# The target is measured in units of $100,000.
print("\nMAE converted to dollars:")
for row in results:
    print(f"{row['Model']}: ${row['MAE'] * 100000:,.2f}")

# STEP 8: Save metrics and predictions
metrics_df.to_csv("svr_metrics.csv", index=False)

prediction_df = pd.DataFrame({
    "Actual_House_Value": y_test.to_numpy()
})

for name, y_pred in predictions.items():
    column_name = name.replace(" ", "_")
    prediction_df[column_name] = y_pred

prediction_df.to_csv("svr_predictions.csv", index=False)

print("\nSaved:")
print("california_housing.csv")
print("svr_metrics.csv")
print("svr_predictions.csv")

# STEP 9: Graph - compare evaluation metrics
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.barplot(data=metrics_df, x="Model", y="MAE", ax=axes[0])
axes[0].set_title("Mean Absolute Error (lower is better)")
axes[0].tick_params(axis="x", rotation=20)

sns.barplot(data=metrics_df, x="Model", y="RMSE", ax=axes[1])
axes[1].set_title("Root Mean Squared Error (lower is better)")
axes[1].tick_params(axis="x", rotation=20)

sns.barplot(data=metrics_df, x="Model", y="R2 Score", ax=axes[2])
axes[2].set_title("R-squared (higher is better)")
axes[2].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.show()

# STEP 10: Graph - actual vs predicted values
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (name, y_pred) in zip(axes, predictions.items()):
    ax.scatter(y_test, y_pred, alpha=0.35)
    ax.set_xlabel("Actual House Value ($100,000 units)")
    ax.set_ylabel("Predicted House Value ($100,000 units)")
    ax.set_title(name + ": Actual vs Predicted")

    # Reference line for perfect predictions
    ax.plot([0, 5], [0, 5], "r--")
    ax.set_xlim(0, 5.2)
    ax.set_ylim(0, 5.2)

plt.tight_layout()
plt.show()

# STEP 11: Graph - prediction errors for RBF model
rbf_predictions = predictions["SVR RBF Kernel"]
errors = y_test.to_numpy() - rbf_predictions

plt.figure(figsize=(8, 5))
sns.histplot(errors, bins=40, kde=True)
plt.xlabel("Actual - Predicted ($100,000 units)")
plt.title("SVR RBF Kernel: Prediction Error Distribution")
plt.tight_layout()
plt.show()

# STEP 12: Show sample predictions from RBF model
sample_results = pd.DataFrame({
    "Actual_Value": y_test.to_numpy()[:10],
    "Predicted_Value": rbf_predictions[:10]
})

sample_results["Actual_Dollars"] = (
    sample_results["Actual_Value"] * 100000
)
sample_results["Predicted_Dollars"] = (
    sample_results["Predicted_Value"] * 100000
)

print("\n========== FIRST 10 PREDICTIONS (RBF) ==========")
print(sample_results.round(3).to_string(index=False))

# STEP 13: Save sample predictions
sample_results.to_csv("svr_sample_predictions.csv", index=False)

# STEP 14: Identify the better model by test RMSE
best_model_row = metrics_df.loc[metrics_df["RMSE"].idxmin()]

print("\n========== BEST MODEL BY TEST RMSE ==========")
print("Model:", best_model_row["Model"])
print(f"RMSE: {best_model_row['RMSE']:.4f}")
print(f"R2 Score: {best_model_row['R2 Score']:.4f}")

print("\nSVR Regression project completed successfully!")
