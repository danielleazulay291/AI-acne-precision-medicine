
from pathlib import Path
import sys
import joblib

# Add project root to Python's import path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.load_data import load_expression_data
from models.model_pipeline import build_pipeline

# Load original dataset
df, metadata = load_expression_data()

X = df.T
sample_metadata = metadata.set_index("sample_id").loc[X.index]
y = (sample_metadata["tissue_type"] == "lesional").astype(int)

print("--- FINAL MODEL TRAINING ---")
print("Training samples:", X.shape[0])
print("Features:", X.shape[1])
print("Class counts:")
print(y.value_counts())

# Build and train the final pipeline
model = build_pipeline()
model.fit(X, y)

# Save trained model
model_path = Path(__file__).resolve().parent / "final_model.joblib"
joblib.dump(model, model_path)

print("\nFinal model trained successfully!")
print("Saved model to:", model_path)
