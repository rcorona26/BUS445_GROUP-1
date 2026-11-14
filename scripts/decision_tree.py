from pathlib import Path
import pandas as pd 
from scipy.sparse import load_npz
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Locate the main project folder
project_folder = Path(__file__).resolve().parent.parent


# Load the training and validation feature matrices
X_train = load_npz(project_folder / "data" / "X_train.npz")
X_val = load_npz(project_folder / "data" / "X_val.npz")

print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)


# Load the training and validation datasets
train_df = pd.read_csv(project_folder / "data" / "train_split.csv")
val_df = pd.read_csv(project_folder / "data" / "validation_split.csv")

# Extract the target labels
y_train = train_df["label"]
y_val = val_df["label"]

# Display the dimensions of our labels
print("Training labels:", y_train.shape)
print("Validation labels:", y_val.shape)

# Create our initial decision tree classifier
tree_model = DecisionTreeClassifier(max_depth=10,random_state=42)

# Train the decision tree using our training data
tree_model.fit(X_train, y_train)

print("Decision tree trained successfully!")


# Predict classifications for the validation dataset
val_predictions = tree_model.predict(X_val)

# Calculate validation accuracy and macro F1-score
val_accuracy = accuracy_score(y_val, val_predictions)
val_macro_f1 = f1_score(y_val, val_predictions, average="macro")

# Display the results
print("\n=== DECISION TREE VALIDATION RESULTS ===")
print("Validation Accuracy:", round(val_accuracy, 3))
print("Validation Macro F1:", round(val_macro_f1, 3))


# Evaluate performance on the training dataset
train_predictions = tree_model.predict(X_train)

train_accuracy = accuracy_score(y_train, train_predictions)
train_macro_f1 = f1_score(y_train, train_predictions, average="macro")

print("\n=== DECISION TREE TRAINING RESULTS ===")
print("Training Accuracy:", round(train_accuracy, 3))
print("Training Macro F1:", round(train_macro_f1, 3))


# Test different maximum tree depths
depth_values = [12, 15, 18, 20, 25]

print("\n=== DECISION TREE DEPTH TUNING ===")

for depth in depth_values:
    model = DecisionTreeClassifier(max_depth=depth,random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(y_val, predictions, average="macro")

    print(f"Depth: {depth} | Accuracy: {accuracy:.3f} | Macro F1: {macro_f1:.3f}")


# Test different minimum leaf sizes
leaf_values = [7, 8, 9, 10, 12, 15]

print("\n=== DECISION TREE LEAF TUNING ===")

for leaf in leaf_values:
    model = DecisionTreeClassifier(max_depth=15,min_samples_leaf=leaf,random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(y_val, predictions, average="macro")

    print(f"Leaf: {leaf} | Accuracy: {accuracy:.3f} | Macro F1: {macro_f1:.3f}")


# Train our selected decision tree
best_tree = DecisionTreeClassifier(max_depth=15,min_samples_leaf=15,random_state=42)
best_tree.fit(X_train, y_train)

# Predict the validation classifications
best_predictions = best_tree.predict(X_val)

# Display performance for each classification
print("\n=== BEST DECISION TREE CLASSIFICATION REPORT ===")
print(classification_report(y_val, best_predictions))


# Train a decision tree with balanced class weights
balanced_tree = DecisionTreeClassifier(max_depth=15,min_samples_leaf=15,class_weight="balanced",random_state=42)

balanced_tree.fit(X_train, y_train)

# Evaluate the balanced tree on validation data
balanced_predictions = balanced_tree.predict(X_val)

balanced_accuracy = accuracy_score(y_val, balanced_predictions)
balanced_macro_f1 = f1_score(y_val, balanced_predictions, average="macro")

print("\n=== BALANCED DECISION TREE RESULTS ===")
print("Validation Accuracy:", round(balanced_accuracy, 3))
print("Validation Macro F1:", round(balanced_macro_f1, 3))
print(classification_report(y_val, balanced_predictions))

# Test different depths while keeping minimum leaf size at 15
depth_values = [8, 10, 12, 15, 18, 20]

print("\n=== DEPTH TUNING WITH LEAF SIZE 15 ===")

for depth in depth_values:
    model = DecisionTreeClassifier(max_depth=depth,min_samples_leaf=15,random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(y_val, predictions, average="macro")

    print(f"Depth: {depth} | Accuracy: {accuracy:.3f} | Macro F1: {macro_f1:.3f}")


# Check training performance of our selected tree
best_train_predictions = best_tree.predict(X_train)

best_train_accuracy = accuracy_score(y_train, best_train_predictions)
best_train_macro_f1 = f1_score(y_train, best_train_predictions, average="macro")

print("\n=== TUNED DECISION TREE TRAINING RESULTS ===")
print("Training Accuracy:", round(best_train_accuracy, 3))
print("Training Macro F1:", round(best_train_macro_f1, 3))


# Load the untouched testing dataset
X_test = load_npz(project_folder / "data" / "X_test.npz")
test_df = pd.read_csv(project_folder / "data" / "test_split.csv")

# Extract the correct testing labels
y_test = test_df["label"]

# Verify that the features and labels match
print("\n=== TEST DATA CHECK ===")
print("Testing features:", X_test.shape)
print("Testing labels:", y_test.shape)

assert X_test.shape[0] == len(y_test), "Testing data mismatch"

# Make predictions using our selected decision tree
test_predictions = best_tree.predict(X_test)

# Calculate final testing performance
test_accuracy = accuracy_score(y_test, test_predictions)
test_macro_f1 = f1_score(y_test, test_predictions, average="macro")

# Display final testing results
print("\n=== FINAL DECISION TREE TEST RESULTS ===")
print("Test Accuracy:", round(test_accuracy, 3))
print("Test Macro F1:", round(test_macro_f1, 3))

# Display performance for each classification
print("\n=== TEST CLASSIFICATION REPORT ===")
print(classification_report(y_test, test_predictions))

# Save our final trained decision tree
model_path = project_folder / "models" / "decision_tree.joblib"

joblib.dump(best_tree, model_path)

print("\nFinal decision tree saved successfully!")

# Create a confusion matrix for the final decision tree
ConfusionMatrixDisplay.from_predictions(y_test,test_predictions,cmap="Blues",values_format="d")

plt.title("Decision Tree - Test Confusion Matrix")
plt.savefig(project_folder / "models" / "decision_tree_confusion_matrix.png",dpi=300, bbox_inches="tight")
plt.show()
