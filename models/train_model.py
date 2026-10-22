
from pathlib import Path
import sys
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Add the project root to Python's import path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.load_data import load_expression_data

# Load gene expression data and metadata
df, metadata = load_expression_data()

print("Expression matrix:", df.shape)
print("Metadata:", metadata.shape)


# Prepare the feature matrix: samples as rows, probes as columns
X = df.T

# Align metadata with the order of samples in X
sample_metadata = metadata.set_index("sample_id").loc[X.index]

# Create binary classification labels
y = (sample_metadata["tissue_type"] == "lesional").astype(int)

# Keep patient IDs for group-aware validation
groups = sample_metadata["patient_id"]

print("\n--- ML DATA PREPARATION ---")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("groups shape:", groups.shape)
print("Class counts:")
print(y.value_counts())

# Create 4-fold cross-validation grouped by patient
cv = GroupKFold(n_splits=4)

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups=groups), start=1
):
    train_patients = set(groups.iloc[train_idx])
    test_patients = set(groups.iloc[test_idx])

    print(f"\nFold {fold}")
    print("Training samples:", len(train_idx))
    print("Validation samples:", len(test_idx))
    print("Training patients:", sorted(train_patients))
    print("Validation patients:", sorted(test_patients))
    print("Patient overlap:", train_patients & test_patients)



# Define the machine learning pipeline

# Define the machine learning pipeline
pipeline = Pipeline([
    ("variance_filter", VarianceThreshold(threshold=0.0)),
    ("feature_selection", SelectKBest(score_func=f_classif, k=100)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        solver="saga",
        l1_ratio=0.5,
        C=0.1,
        max_iter=5000,
        random_state=42
    ))
])

print("\n--- ML PIPELINE ---")
print(pipeline)

print("\n--- MODEL TRAINING ---")

fold_accuracies = []
all_true = []
all_pred = []

for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups=groups), start=1):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = clone(pipeline)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    all_true.extend(y_test.tolist())
    all_pred.extend(y_pred.tolist())

    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)

    print(f"\nFold {fold}")
    print("Actual labels:", y_test.tolist())
    print("Predicted labels:", y_pred.tolist())
    print(f"Accuracy: {accuracy:.2%}")


print("\n--- OVERALL MODEL EVALUATION ---")

print("Mean Accuracy:", sum(fold_accuracies) / len(fold_accuracies))

print("\nClassification Report:")
print(classification_report(
    all_true,
    all_pred,
    target_names=["Non-lesional", "Lesional"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(all_true, all_pred))




