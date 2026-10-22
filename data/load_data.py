from pathlib import Path
import gzip
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt


def load_expression_data():
    """Load the RMA-normalized GSE53795 expression matrix and sample metadata."""

    file_path = Path(__file__).parent / "GSE53795_series_matrix.txt.gz"

    with gzip.open(file_path, "rt", encoding="utf-8") as file:
        lines = file.readlines()

    start = next(
        i for i, line in enumerate(lines)
        if line.startswith("!series_matrix_table_begin")
    )

    end = next(
        i for i, line in enumerate(lines)
        if line.startswith("!series_matrix_table_end")
    )

    matrix_text = "".join(lines[start + 1:end])

    df = pd.read_csv(StringIO(matrix_text), sep="\t", index_col=0)

    # Data are already RMA-normalized (log2 scale).
    # No additional RMA normalization or log2 transformation is needed.

    def get_metadata(field_name):
        for line in lines:
            if line.startswith(field_name + "\t"):
                return [
                    value.strip('"')
                    for value in line.strip().split("\t")[1:]
                ]
        raise ValueError(f"Metadata field not found: {field_name}")

    sample_ids = get_metadata("!Sample_geo_accession")
    sample_titles = get_metadata("!Sample_title")

    metadata = pd.DataFrame({
        "sample_id": sample_ids,
        "sample_title": sample_titles
    })

    metadata["patient_id"] = (
        metadata["sample_title"]
        .str.extract(r"Pt(\d+)")[0]
        .astype(int)
    )

    metadata["tissue_type"] = metadata["sample_title"].apply(
        lambda title: "non-lesional"
        if "non lesional" in title.lower()
        else "lesional"
    )

    return df, metadata

if __name__ == "__main__":
    df, metadata = load_expression_data()
    print("\n--- SAMPLE METADATA TABLE ---")
    print(metadata.to_string(index=False))

    print("\nTissue counts:")
    print(metadata["tissue_type"].value_counts())

    print("\nNumber of unique patients:")
    print(metadata["patient_id"].nunique())


    print("\n--- DATA QUALITY CONTROL ---")

    print("Matrix shape:", df.shape)

    print("Missing values:", df.isna().sum().sum())

    print("Data types:")
    print(df.dtypes.value_counts())

    print("Minimum expression:", df.min().min())
    print("Maximum expression:", df.max().max())

    print("Sample IDs match metadata:",
      set(df.columns) == set(metadata["sample_id"]))

    print("Duplicate probe IDs:", df.index.duplicated().sum())


    # Compare expression distributions across samples
    plt.figure(figsize=(14, 6))

    df.boxplot(rot=90, grid=False)

    plt.title("Gene Expression Distribution Across Samples")
    plt.xlabel("GEO Sample ID")
    plt.ylabel("Expression Value")
    plt.tight_layout()

    plt.show()








