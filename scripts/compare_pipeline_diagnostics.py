#!/usr/bin/env python3
"""Generate diagnostic plots for baseline vs optimized search results.

This script loads the two CSV result files from the project:
- results/clusters_v3.csv
- results/max_docs.csv

It generates three PNG figures to diagnose the recall/latency tradeoff and
how the optimized search path differs internally.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"


def load_csv(path: Path) -> pd.DataFrame:
    """Load and normalize a CSV result file."""
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    df = pd.read_csv(path)

    numeric_cols = [
        "k", "itr", "nb", "nd", "md",
        "Avg_Cluster_Size", "Median_Cluster_Size", "Avg_Cluster_Similarity",
        "Avg_Blocks_Entered", "Avg_Blocks_Skipped", "Avg_Docs_Examined",
        "Avg_Docs_Popped", "Clustering_Time_s", "Indexing_Time_s",
        "Recall@30", "Avg_Time_Per_Query_ms", "Search_Time_s", "Total_Time_s",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def summarize_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact summary of average metric values per dataset."""
    summary = df.groupby("dataset", as_index=False).agg(
        Avg_Recall=("Recall@30", "mean"),
        Avg_Latency=("Avg_Time_Per_Query_ms", "mean"),
        Avg_Docs_Examined=("Avg_Docs_Examined", "mean"),
        Avg_Blocks_Entered=("Avg_Blocks_Entered", "mean"),
        Avg_Blocks_Skipped=("Avg_Blocks_Skipped", "mean"),
        Avg_Search_Time=("Search_Time_s", "mean"),
        Count=("Recall@30", "size"),
    )
    return summary


def plot_recall_vs_latency(baseline_df: pd.DataFrame, optimized_df: pd.DataFrame) -> None:
    """Plot Recall@30 vs Avg_Time_Per_Query_ms for both datasets."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        baseline_df["Avg_Time_Per_Query_ms"],
        baseline_df["Recall@30"],
        s=90,
        c="#1f77b4",
        marker="o",
        edgecolor="k",
        alpha=0.85,
        label="Baseline (clusters_v3.csv)",
    )
    ax.scatter(
        optimized_df["Avg_Time_Per_Query_ms"],
        optimized_df["Recall@30"],
        s=90,
        c="#d62728",
        marker="^",
        edgecolor="k",
        alpha=0.9,
        label="Optimized (max_docs.csv)",
    )

    ax.axhline(0.90, color="black", linestyle="--", linewidth=2, label="Recall@30 = 0.90")

    ax.set_xlabel("Avg_Time_Per_Query_ms", fontsize=11)
    ax.set_ylabel("Recall@30", fontsize=11)
    ax.set_title("Recall vs. Latency Tradeoff Across Pipeline Variants", fontsize=13, weight="bold")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_recall_vs_latency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_search_pruning_diagnosis(baseline_df: pd.DataFrame, optimized_df: pd.DataFrame) -> None:
    """Compare mean search-loop internals between datasets."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    metrics = [
        ("Avg_Docs_Examined", "Documents Examined"),
        ("Avg_Blocks_Entered", "Blocks Entered"),
        ("Avg_Blocks_Skipped", "Blocks Skipped"),
    ]

    summary = summarize_by_dataset(
        pd.concat(
            [
                baseline_df.assign(dataset="Baseline"),
                optimized_df.assign(dataset="Optimized"),
            ],
            ignore_index=True,
        )
    )

    for ax, (metric, label) in zip(axes, metrics):
        plot_df = pd.DataFrame(
            {
                "dataset": summary["dataset"],
                metric: summary[metric],
                "label": label,
            }
        )

        sns.barplot(
            data=plot_df,
            x="dataset",
            y=metric,
            hue="dataset",
            palette=["#1f77b4", "#d62728"],
            ax=ax,
            dodge=False,
            legend=False,
        )
        ax.set_title(f"Mean {label}")
        ax.set_ylabel("Mean Value")
        ax.set_xlabel("Pipeline Variant")
        ax.tick_params(axis="x", rotation=0)
        ax.legend([], [], frameon=False)

    fig.suptitle("Search Pruning and Execution Diagnosis", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES_DIR / "02_search_pruning_diagnosis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_k_effect_on_baseline(df: pd.DataFrame) -> None:
    """Show how increasing k affects recall and latency in clusters_v3.csv."""
    plt.style.use("seaborn-v0_8-whitegrid")

    summary = (
        df.groupby("k", as_index=False)
        .agg(
            Recall_Mean=("Recall@30", "mean"),
            Recall_Std=("Recall@30", "std"),
            Latency_Mean=("Avg_Time_Per_Query_ms", "mean"),
            Latency_Std=("Avg_Time_Per_Query_ms", "std"),
            Search_Time_Mean=("Search_Time_s", "mean"),
        )
        .sort_values("k")
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=False)

    axes[0].errorbar(
        summary["k"],
        summary["Recall_Mean"],
        yerr=summary["Recall_Std"],
        fmt="o-",
        color="#2ca02c",
        linewidth=2,
        capsize=4,
        label="Mean Recall@30",
    )
    axes[0].axhline(0.90, color="black", linestyle="--", linewidth=1.5, label="Target 0.90")
    axes[0].set_title("Effect of Increasing k on Recall@30")
    axes[0].set_xlabel("k (number of clusters)")
    axes[0].set_ylabel("Recall@30")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="best")

    axes[1].errorbar(
        summary["k"],
        summary["Latency_Mean"],
        yerr=summary["Latency_Std"],
        fmt="o-",
        color="#ff7f0e",
        linewidth=2,
        capsize=4,
        label="Mean Avg_Time_Per_Query_ms",
    )
    axes[1].set_title("Effect of Increasing k on Query Latency")
    axes[1].set_xlabel("k (number of clusters)")
    axes[1].set_ylabel("Avg_Time_Per_Query_ms")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(loc="best")

    fig.suptitle("Baseline clusters_v3.csv: Effect of Increasing k", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES_DIR / "04_k_effect_clusters_v3.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_max_docs_parameter_heatmaps(df: pd.DataFrame) -> None:
    """Show how nb and nd affect recall and latency for the optimized search run."""
    plt.style.use("seaborn-v0_8-whitegrid")

    # Create a clean matrix for nb vs nd.
    recall_matrix = df.pivot_table(index="nb", columns="nd", values="Recall@30", aggfunc="mean")
    latency_matrix = df.pivot_table(index="nb", columns="nd", values="Avg_Time_Per_Query_ms", aggfunc="mean")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=False)

    sns.heatmap(
        recall_matrix,
        ax=axes[0],
        cmap="RdYlGn",
        vmin=0.30,
        vmax=0.95,
        annot=True,
        fmt=".3f",
        linewidths=0.25,
        cbar_kws={"label": "Recall@30"},
    )
    axes[0].set_title("Recall@30 Heatmap (max_docs.csv)")
    axes[0].set_xlabel("nd (documents per block)")
    axes[0].set_ylabel("nb (blocks per dimension)")

    sns.heatmap(
        latency_matrix,
        ax=axes[1],
        cmap="YlOrRd",
        vmin=latency_matrix.min().min(),
        vmax=latency_matrix.max().max(),
        annot=True,
        fmt=".2f",
        linewidths=0.25,
        cbar_kws={"label": "Avg_Time_Per_Query_ms"},
    )
    axes[1].set_title("Avg Time per Query Heatmap (max_docs.csv)")
    axes[1].set_xlabel("nd (documents per block)")
    axes[1].set_ylabel("nb (blocks per dimension)")

    fig.suptitle("Impact of nb and nd on the Optimized Search Variant", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES_DIR / "03_nb_nd_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)

    try:
        baseline_df = load_csv(RESULTS_DIR / "clusters_v3.csv").copy()
        optimized_df = load_csv(RESULTS_DIR / "max_docs.csv").copy()
        baseline_df["dataset"] = "baseline"
        optimized_df["dataset"] = "optimized"

        plot_recall_vs_latency(baseline_df, optimized_df)
        plot_search_pruning_diagnosis(baseline_df, optimized_df)
        plot_k_effect_on_baseline(baseline_df)
        plot_max_docs_parameter_heatmaps(optimized_df)

        print("Generated:")
        print("  -", FIGURES_DIR / "01_recall_vs_latency.png")
        print("  -", FIGURES_DIR / "02_search_pruning_diagnosis.png")
        print("  -", FIGURES_DIR / "04_k_effect_clusters_v3.png")
        print("  -", FIGURES_DIR / "03_nb_nd_heatmaps.png")

    except Exception as exc:
        raise RuntimeError(f"Failed to generate plots: {exc}") from exc


if __name__ == "__main__":
    main()
