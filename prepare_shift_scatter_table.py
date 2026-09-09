# import os
# import pandas as pd

# INPUT_CSV = "outputs/real_experiment/all_batches_results.csv"
# OUTPUT_CSV = "outputs/real_experiment/shift_vs_selection_loss.csv"

# df = pd.read_csv(INPUT_CSV)

# SOURCE_SUFFIX = {
#     "ArXiv": "ArXiv",
#     "FreeLaw": "FreeLaw",
#     "PubMed Central": "PubMed_Central",
# }

# SHIFT_METRICS = [
#     "token_kl",
#     "embedding_mean_l2",
#     "diag_gaussian_kl",
#     "pca_gaussian_kl_5",
#     "pca_gaussian_kl_10",
#     "pca_gaussian_kl_20",
# ]


# def get_selected_source_shift(row, metric):
#     source = SOURCE_SUFFIX[row["selected_source"]]
#     col = f"batch_to_source_{metric}_{source}"
#     return row[col]


# out = pd.DataFrame()

# out["deployment_dataset"] = df["deployment_dataset"]

# if "batch_id" in df.columns:
#     out["batch_id"] = df["batch_id"]
# elif "batch_idx" in df.columns:
#     out["batch_id"] = df["batch_idx"]
# elif "selection_batch_id" in df.columns:
#     out["batch_id"] = df["selection_batch_id"]
# else:
#     out["batch_id"] = range(len(df))

# out["selected_source"] = df["selected_source"]
# out["selection_loss"] = df["algo2_selection_loss"]

# for metric in SHIFT_METRICS:
#     out[metric] = df.apply(
#         lambda row: get_selected_source_shift(row, metric),
#         axis=1,
#     )

# os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
# out.to_csv(OUTPUT_CSV, index=False)

# print(f"Saved: {OUTPUT_CSV}")
# print()
# print(out.head())


import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = "outputs/real_experiment/shift_vs_selection_loss.csv"
OUTPUT_DIR = "outputs/real_experiment/shift_vs_selection_loss_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

SHIFT_METRICS = {
    "token_kl": "Token KL",
    "embedding_mean_l2": "Embedding Mean L2",
    "diag_gaussian_kl": "Diagonal Gaussian KL",
    "pca_gaussian_kl_5": "PCA Gaussian KL (5D)",
    "pca_gaussian_kl_10": "PCA Gaussian KL (10D)",
    "pca_gaussian_kl_20": "PCA Gaussian KL (20D)",
}

# One category = deployment dataset -> selected source
df["combination"] = (
    df["deployment_dataset"]
    + " → "
    + df["selected_source"]
)

# Fixed colors across ALL figures
COLORS = {
    "PG-19 → FreeLaw": "#1f77b4",
    "PubMed Abstracts → PubMed Central": "#2ca02c",
    "GitHub → ArXiv": "#d62728",
    "GitHub → FreeLaw": "#ff7f0e",
    "Ubuntu IRC → PubMed Central": "#9467bd",
    "Ubuntu IRC → ArXiv": "#8c564b",
}

# Keep a fixed legend order
COMBINATION_ORDER = [
    "PG-19 → FreeLaw",
    "PubMed Abstracts → PubMed Central",
    "GitHub → ArXiv",
    "GitHub → FreeLaw",
    "Ubuntu IRC → PubMed Central",
    "Ubuntu IRC → ArXiv",
]

for metric, metric_label in SHIFT_METRICS.items():

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for combination in COMBINATION_ORDER:
        subset = df[df["combination"] == combination]

        if subset.empty:
            continue

        ax.scatter(
            subset[metric],
            subset["selection_loss"],
            s=65,
            alpha=0.85,
            color=COLORS[combination],
            label=combination,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_xlabel(metric_label, fontsize=12)
    ax.set_ylabel("Selection Loss", fontsize=12)

    ax.set_title(
        f"Selection Loss vs. {metric_label}",
        fontsize=13,
    )

    ax.grid(
        linestyle="--",
        alpha=0.3,
    )

    ax.legend(
        title="Deployment batch → Selected source",
        fontsize=9,
        title_fontsize=9,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{metric}_vs_selection_loss.png",
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")

print("\nAll 6 scatter plots generated.")