
from pathlib import Path
import sys

import joblib
import pandas as pd

# Project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.load_data import load_expression_data

# Load trained model
model_path = project_root / "models" / "final_model.joblib"
model = joblib.load(model_path)

# Load original feature names
expression_df, metadata = load_expression_data()
all_probes = expression_df.index.to_numpy()

# Retrieve fitted pipeline components
variance_filter = model.named_steps["variance_filter"]
feature_selection = model.named_steps["feature_selection"]
classifier = model.named_steps["classifier"]

# Track probes through both selection steps
probes_after_variance = all_probes[variance_filter.get_support()]
selected_probes = probes_after_variance[feature_selection.get_support()]

# Logistic regression coefficients
coefficients = classifier.coef_[0]

assert len(selected_probes) == len(coefficients)

# Create results table
results = pd.DataFrame({
    "probe_id": selected_probes,
    "coefficient": coefficients,
})

results["absolute_coefficient"] = results["coefficient"].abs()
results = results.sort_values(
    "absolute_coefficient", ascending=False
)

# Save results
output_path = project_root / "results" / "selected_probes.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output_path, index=False)

print("Selected probes:", len(results))
print("\nTop 20 probes:")
print(results.head(20).to_string(index=False))
print("\nResults saved to:", output_path)
