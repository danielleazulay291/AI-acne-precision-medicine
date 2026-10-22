
from pathlib import Path
import sys
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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
pipeline = Pipeline([
    ("variance_filter", VarianceThreshold(threshold=0.0)),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=0.1,
        max_iter=5000,
        random_state=42
    ))
])

print("\n--- ML PIPELINE ---")
print(pipeline)


