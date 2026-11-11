from pathlib import Path
from scipy.sparse import load_npz
import pandas as pd

# Locate the main project folder
project_folder = Path(__file__).resolve().parent.parent

# Load the saved TF-IDF feature matrices
X_train = load_npz(project_folder / "data" / "X_train.npz")
X_val = load_npz(project_folder / "data" / "X_val.npz")
X_test = load_npz(project_folder / "data" / "X_test.npz")

# Display the dimensions of the loaded matrices
print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)
print("Testing features:", X_test.shape)

# Load the datasets containing our classification labels
train_df = pd.read_csv(project_folder / "data" / "train_split.csv")
val_df = pd.read_csv(project_folder / "data" / "validation_split.csv")
test_df = pd.read_csv(project_folder / "data" / "test_split.csv")

# Extract the target labels
y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]

# Confirm that features and labels have matching row counts
print("\nTraining labels:", y_train.shape)
print("Validation labels:", y_val.shape)
print("Testing labels:", y_test.shape)


# Verify that each feature matrix matches its labels
assert X_train.shape[0] == len(y_train), "Training data mismatch"
assert X_val.shape[0] == len(y_val), "Validation data mismatch"
assert X_test.shape[0] == len(y_test), "Testing data mismatch"

# Verify that all datasets use the same number of features
assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], \
    "Feature count mismatch"

print("\nAll data consistency checks passed!")
