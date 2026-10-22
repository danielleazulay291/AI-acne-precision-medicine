from pathlib import Path
import sys
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.load_data import load_expression_data
from data.load_external_data import (
    external_X,
    external_y,
    selected_metadata
)


# Load original training dataset
original_df, original_metadata = load_expression_data()

# Load external dataset
external_file = (
    project_root / "data" / "GSE108110_series_matrix.txt.gz"
)

external_df = pd.read_csv(
    external_file,
    sep="\t",
    comment="!",
    index_col=0
)

# Compare gene expression distributions
print("\n--- DATASET DISTRIBUTION COMPARISON ---")

for name, dataset in [
    ("Original", original_df),
    ("External", external_df)
]:
    values = dataset.to_numpy().flatten()

    print(f"\n{name} dataset:")
    print(f"Minimum: {values.min():.2f}")
    print(f"Maximum: {values.max():.2f}")
    print(f"Mean: {values.mean():.2f}")
    print(f"Median: {pd.Series(values).median():.2f}")



print("\n--- EXTERNAL VALIDATION ---")

# Load the previously trained model
model_path = project_root / "models" / "final_model.joblib"
model = joblib.load(model_path)

# Ensure external probes match the training feature order
external_X = external_X.loc[:, original_df.index]

# Validate external data before prediction
assert external_X.shape == (54, 54675)
assert external_X.index.tolist() == selected_metadata["sample_id"].tolist()
assert not external_X.isna().any().any()

# Predict without fitting on external data
predictions = model.predict(external_X)


# Compare predictions with the known labels
print("\nExternal Accuracy:",
      accuracy_score(external_y, predictions))

print("\nClassification Report:")
print(classification_report(
    external_y,
    predictions,
    target_names=["Non-lesional", "Lesional"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    external_y,
    predictions,
    labels=[0, 1]
))

# Evaluate predictions for each biological group
results = selected_metadata.copy()
results["actual_label"] = external_y.to_numpy()
results["predicted_label"] = predictions
results["correct"] = (
    results["actual_label"] == results["predicted_label"]
)

print("\n--- RESULTS BY BIOLOGICAL GROUP ---")
print(
    results.groupby("biological_group")["correct"]
    .agg(["count", "sum", "mean"])
    .to_string()
)
