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
