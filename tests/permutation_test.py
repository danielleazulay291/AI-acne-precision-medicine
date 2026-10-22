from pathlib import Path
import sys

import numpy as np

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.load_data import load_expression_data

# Load the same dataset used in model training
df, metadata = load_expression_data()

X = df.T
sample_metadata = metadata.set_index("sample_id").loc[X.index]

y = (sample_metadata["tissue_type"] == "lesional").astype(int)
groups = sample_metadata["patient_id"]

cv = GroupKFold(n_splits=4)

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


def evaluate_model(labels):
    predictions = np.empty(len(labels), dtype=int)

    for train_idx, test_idx in cv.split(X, labels, groups=groups):
        model = clone(pipeline)

        model.fit(X.iloc[train_idx], labels[train_idx])
        predictions[test_idx] = model.predict(X.iloc[test_idx])

    return accuracy_score(labels, predictions)


# Original labels
original_y = y.to_numpy()
original_accuracy = evaluate_model(original_y)

# Permutation test
rng = np.random.default_rng(42)

n_permutations = 20
permuted_accuracies = []

patient_groups = groups.to_numpy()

for i in range(n_permutations):
    permuted_y = original_y.copy()

    # Randomly swap labels within each patient
    for patient_id in np.unique(patient_groups):
        patient_indices = np.flatnonzero(
            patient_groups == patient_id
        )

        if rng.random() < 0.5:
            permuted_y[patient_indices] = permuted_y[
                patient_indices[::-1]
            ]

    accuracy = evaluate_model(permuted_y)
    permuted_accuracies.append(accuracy)

    print(
        f"Permutation {i + 1}/{n_permutations}: "
        f"Accuracy = {accuracy:.2%}"
    )


# Calculate empirical p-value
permuted_accuracies = np.array(permuted_accuracies)

p_value = (
    1 + np.sum(permuted_accuracies >= original_accuracy)
) / (n_permutations + 1)


print("\n--- PERMUTATION TEST RESULTS ---")
print(f"Original Accuracy: {original_accuracy:.2%}")
print(f"Mean Permuted Accuracy: {permuted_accuracies.mean():.2%}")
print(f"Maximum Permuted Accuracy: {permuted_accuracies.max():.2%}")
print(f"Empirical p-value: {p_value:.4f}")

print("\n--- SAMPLE METADATA CHECK ---")

print(sample_metadata[
    ["patient_id", "tissue_type"]
].to_string())

print("\n--- AVAILABLE METADATA COLUMNS ---")
print(metadata.columns.tolist())

print("\n--- METADATA PREVIEW ---")
print(metadata.head().to_string())