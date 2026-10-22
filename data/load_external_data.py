from pathlib import Path
import gzip
import csv
import pandas as pd
import sys

file_path = Path(__file__).resolve().parent / "GSE108110_series_matrix.txt.gz"

with gzip.open(file_path, "rt", encoding="utf-8") as file:
    for line in file:
        if line.startswith("!series_matrix_table_begin"):
            break

    df = pd.read_csv(file, sep="\t", index_col=0)

df = df.drop(index="!series_matrix_table_end", errors="ignore")
df = df.apply(pd.to_numeric, errors="coerce")

print("--- EXTERNAL DATASET ---")
print("Dataset shape:", df.shape)
print("Number of probes:", len(df.index))
print("Number of samples:", len(df.columns))
print("Missing values:", df.isna().sum().sum())
print("\nFirst 5 rows:")
print(df.iloc[:5, :5])


sample_info = {}

with gzip.open(file_path, "rt", encoding="utf-8") as file:
    for line in file:
        if line.startswith("!Sample_geo_accession") or \
           line.startswith("!Sample_title") or \
           line.startswith("!Sample_source_name_ch1"):

            values = next(csv.reader([line], delimiter="\t"))
            sample_info[values[0]] = values[1:]

        if line.startswith("!series_matrix_table_begin"):
            break

metadata = pd.DataFrame({
    "sample_id": sample_info["!Sample_geo_accession"],
    "sample_title": sample_info["!Sample_title"],
    "source": sample_info["!Sample_source_name_ch1"]
})

print("\n--- EXTERNAL SAMPLE TYPES ---")
print(metadata["source"].value_counts())

print("\n--- SAMPLE METADATA PREVIEW ---")
print(metadata.head(12).to_string(index=False))

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.load_data import load_expression_data

original_df, original_metadata = load_expression_data()

print("\n--- PROBE COMPATIBILITY CHECK ---")

print("Original probes:", len(original_df.index))
print("External probes:", len(df.index))

same_probes = original_df.index.equals(df.index)

print("Identical probe IDs and order:", same_probes)

common_probes = original_df.index.intersection(df.index)

print("Common probes:", len(common_probes))


# -----------------------------------------
# PREPARING EXTERNAL VALIDATION DATA
# -----------------------------------------

print("\n--- PREPARING EXTERNAL VALIDATION DATA ---")

# Keep all 54 samples
selected_metadata = metadata.copy()

# Preserve the original biological group
selected_metadata["biological_group"] = selected_metadata["source"]

# Define binary classification labels
selected_metadata["tissue_type"] = "lesional"

selected_metadata.loc[
    selected_metadata["source"].str.contains(
        "non-lesional skin", case=False, na=False
    ),
    "tissue_type"
] = "non-lesional"

# Create numeric labels
external_y = (
    selected_metadata["tissue_type"] == "lesional"
).astype(int)

# Samples as rows, probes as columns
external_X = df[selected_metadata["sample_id"]].T

print("Total samples:", len(selected_metadata))

print("\nBinary class distribution:")
print(selected_metadata["tissue_type"].value_counts())

print("\nBiological groups:")
print(selected_metadata["biological_group"].value_counts())

print("\nExternal X shape:", external_X.shape)
print("External y shape:", external_y.shape)

print("\nSample order matches metadata:",
      external_X.index.tolist() == selected_metadata["sample_id"].tolist())

