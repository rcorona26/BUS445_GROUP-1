from pathlib import Path
import pandas as pd
import joblib
from scipy.sparse import load_npz
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# Locate the main project folder
project_folder = Path(__file__).resolve().parent.parent

# Load training and validation feature matrices
X_train = load_npz(project_folder / "data" / "X_train.npz")
X_val = load_npz(project_folder / "data" / "X_val.npz")

# Load training and validation labels
train_df = pd.read_csv(project_folder / "data" / "train_split.csv")
val_df = pd.read_csv(project_folder / "data" / "validation_split.csv")

y_train = train_df["label"]
y_val = val_df["label"]

# Verify that the data loaded correctly
print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)
print("Training labels:", y_train.shape)
print("Validation labels:", y_val.shape)

# # Scale the features for neural network training
# scaler = StandardScaler(with_mean=False)

# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)

# Create our baseline neural network
nn_model = MLPClassifier(
    hidden_layer_sizes=(15,),
    activation="logistic",
    solver="lbfgs",
    alpha=0.2,
    max_iter=500,
    random_state=42
)



# Train the neural network using original features
nn_model.fit(X_train, y_train)

# Generate predictions on original validation features
val_predictions = nn_model.predict(X_val)


# Evaluate the model
val_accuracy = accuracy_score(y_val, val_predictions)
val_macro_f1 = f1_score(y_val, val_predictions, average="macro")

print("\nBaseline Neural Network Results")
print(f"Validation Accuracy: {val_accuracy:.3f}")
print(f"Validation Macro F1: {val_macro_f1:.3f}")

# Define hyperparameter values to test
sizes = [5, 10, 15, 20]
alphas = [0.2, 0.4, 0.6]

print("\nNeural Network Hyperparameter Values")
print("Hidden layer sizes:", sizes)
print("Alpha values:", alphas)


# # Test different neural network hyperparameters
# for size in sizes:
#     for alpha in alphas:

#         # Create a neural network with these parameters
#         model = MLPClassifier(
#             hidden_layer_sizes=(size,),
#             activation="logistic",
#             solver="lbfgs",
#             alpha=alpha,
#             max_iter=300,
#             random_state=42
#         )

#         # Train the model
#         model.fit(X_train, y_train)

#         # Predict validation classifications
#         predictions = model.predict(X_val)

#         # Calculate validation performance
#         accuracy = accuracy_score(y_val, predictions)
#         macro_f1 = f1_score(y_val, predictions, average="macro")

#         print(
#             f"Size: {size}, Alpha: {alpha}, "
#             f"Accuracy: {accuracy:.3f}, Macro F1: {macro_f1:.3f}"
#         )


# Load the testing dataset
X_test = load_npz(project_folder / "data" / "X_test.npz")

test_df = pd.read_csv(project_folder / "data" / "test_split.csv")
y_test = test_df["label"]

# Verify testing features match testing labels
print("\nTesting features:", X_test.shape)
print("Testing labels:", y_test.shape)

assert X_test.shape[0] == len(y_test), "Testing data mismatch"
assert X_test.shape[1] == X_train.shape[1], "Feature count mismatch"

# Generate predictions on the testing dataset
test_predictions = nn_model.predict(X_test)

# Calculate testing performance
test_accuracy = accuracy_score(y_test, test_predictions)
test_macro_f1 = f1_score(y_test, test_predictions, average="macro")

# Display final testing results
print("\nFinal Neural Network Test Results")
print(f"Test Accuracy: {test_accuracy:.3f}")
print(f"Test Macro F1: {test_macro_f1:.3f}")

# Display precision, recall, F1, and support for each class
print("\nNeural Network Classification Report")
print(classification_report(y_test, test_predictions))

# Generate neural network confusion matrix
labels = ["Good Fit", "No Fit", "Potential Fit"]

ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_predictions,
    labels=labels,
    cmap="Blues",
    values_format="d"
)

plt.title("Neural Network - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    project_folder / "models" / "neural_network_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight")
plt.show()

# Save our final trained neural network
model_path = project_folder / "models" / "neural_network.joblib"

joblib.dump(nn_model, model_path)

print("\nFinal neural network saved successfully!")
