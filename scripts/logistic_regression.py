import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Locate the main project folder
project_folder = Path(__file__).resolve().parent.parent

# Load the TF-IDF feature matrices
X_train = load_npz(project_folder / "data" / "X_train.npz")
X_val = load_npz(project_folder / "data" / "X_val.npz")
X_test = load_npz(project_folder / "data" / "X_test.npz")

# Load the labels
train_df = pd.read_csv(project_folder / "data" / "train_split.csv")
val_df = pd.read_csv(project_folder / "data" / "validation_split.csv")
test_df = pd.read_csv(project_folder / "data" / "test_split.csv")

y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]

# print("Training features:", X_train.shape)
# print("Validation features:", X_val.shape)
# print("Test features:", X_test.shape)

# Baseline: the percentage our logistic regression model has to beat
most_common_label = y_train.value_counts().idxmax()
baseline_accuracy = (y_test == most_common_label).mean()

# print ("\nMost common label in training set:", most_common_label)
# print ("Percentage to beat:", round(baseline_accuracy * 100, 1), "%\n")

#Try different C values to find the best one
c_values = [0.1, 1, 10, 100]
best_c = None
best_val_f1 = 0

print("\nTuning results (validation set): ")
for C in c_values:
    model = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)
    val_f1 = f1_score(y_val, val_predictions, average="macro")
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"C= {C:<6} Accuracy = {val_accuracy:.3f} Macro F1 = {val_f1:.3f}")

    if val_f1 > best_val_f1:
        best_C = C
        best_val_f1 = val_f1

print("\nBest C:", best_C)

# Train the final model with the best C and test it once
final_model = LogisticRegression(C=best_C, max_iter=2000, class_weight="balanced")
final_model.fit(X_train, y_train)
test_predictions = final_model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions)
test_macro_f1 = f1_score(y_test, test_predictions, average="macro")

print("\n===== FINAL TEST RESULTS =====")
print("Test Accuracy:", round(test_accuracy,3))
print("Test Macro F1:", round(test_macro_f1,3))
print("Beats baseline?", test_accuracy > baseline_accuracy)