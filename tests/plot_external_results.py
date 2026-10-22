
from pathlib import Path
import matplotlib.pyplot as plt

# Project directory
project_root = Path(__file__).resolve().parent.parent

# External validation results: GSE108110
groups = [
    "Non-scar / Non-lesional",
    "Non-scar / Day 21",
    "Non-scar / Early papule",
    "Scar-prone / Non-lesional",
    "Scar-prone / Day 21",
    "Scar-prone / Early papule",
]

correct = [7, 3, 9, 9, 6, 9]
total = [9, 9, 9, 9, 9, 9]

accuracy = [
    100 * c / t
    for c, t in zip(correct, total)
]

fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.barh(groups, accuracy)

for bar, c, t in zip(bars, correct, total):
    ax.text(
        bar.get_width() + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{c}/{t} ({100 * c / t:.1f}%)",
        va="center",
        fontsize=10
    )

ax.set_xlim(0, 115)
ax.set_xlabel("Classification Accuracy (%)")
ax.set_title(
    "External Validation — GSE108110\n"
    "Performance by Biological Group"
)

ax.invert_yaxis()

plt.tight_layout()

# Save chart
output_path = (
    project_root / "results" /
    "external_validation_chart.png"
)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("Chart saved to:", output_path)
