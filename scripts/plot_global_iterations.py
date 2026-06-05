#!/usr/bin/env python3
"""Generate plots from a global iterations CSV.

Creates two plots for a fixed `k` and `itr`:
- Scatter of `nd` vs `nb` colored by `Avg_Time_Per_Query_ms` (color range 0.1ms -> 100ms)
- Scatter of `nd` vs `nb` colored red/green by `Recall@30` (<0.9 red, >=0.9 green)

Usage:
    python scripts/plot_global_iterations.py --csv results/global_iterations_test.csv --k 1500 --itr 3 --outdir figures

"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def read_and_filter(csv_path, k, itr):
    df = pd.read_csv(csv_path)
    # Ensure numeric types for key columns
    for col in ["k", "itr", "nd", "nb", "Avg_Time_Per_Query_ms", "Recall@30"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["k", "itr", "nd", "nb"]) if not df.empty else df
    df_f = df[(df["k"] == k) & (df["itr"] == itr)]
    return df_f


def plot_time(df, k, itr, outpath):
    if df.empty:
        raise ValueError("No rows to plot for the requested k/itr")

    x = df["nd"]
    y = df["nb"]
    c = df["Avg_Time_Per_Query_ms"] if "Avg_Time_Per_Query_ms" in df.columns else None

    fig, ax = plt.subplots(figsize=(8, 6))

    # Map times into color range 0.1 ms (best) -> 100 ms (worst).
    # Use a reversed Red-Yellow-Green so low times are green, high times red.
    cmap = plt.get_cmap("RdYlGn_r")
    # Use log normalization to cover wide range; clamp to [0.1, 100]
    norm = mcolors.LogNorm(vmin=0.1, vmax=100)

    sc = ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=100, edgecolor="k")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Avg_Time_Per_Query_ms")

    ax.set_xlabel("nd")
    ax.set_ylabel("nb")
    ax.set_title(f"Avg time per query (ms) — k={k}, itr={itr}")

    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def plot_recall(df, k, itr, outpath):
    if df.empty:
        raise ValueError("No rows to plot for the requested k/itr")

    x = df["nd"]
    y = df["nb"]
    r = df["Recall@30"] if "Recall@30" in df.columns else None

    colors = ["green" if (val is not None and val >= 0.9) else "red" for val in r]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, c=colors, s=100, edgecolor="k")

    # Legend
    import matplotlib.patches as mpatches

    green = mpatches.Patch(color="green", label="Recall >= 0.9")
    red = mpatches.Patch(color="red", label="Recall < 0.9")
    ax.legend(handles=[green, red])

    ax.set_xlabel("nd")
    ax.set_ylabel("nb")
    ax.set_title(f"Recall@30 — k={k}, itr={itr}")

    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot global iterations results")
    p.add_argument("--csv", required=True, help="Path to global iterations CSV")
    p.add_argument("--k", required=True, type=float, help="Value of k to filter (integer)")
    p.add_argument("--itr", required=True, type=float, help="Value of itr to filter (integer)")
    p.add_argument("--outdir", default="figures", help="Output directory for plots")
    args = p.parse_args()

    csv_path = args.csv
    k = args.k
    itr = args.itr
    outdir = args.outdir

    os.makedirs(outdir, exist_ok=True)

    df = read_and_filter(csv_path, k, itr)
    if df.empty:
        print(f"No data found in {csv_path} for k={k} and itr={itr}")
        sys.exit(1)

    time_out = os.path.join(outdir, f"global_iterations_k{int(k)}_itr{int(itr)}_time.png")
    recall_out = os.path.join(outdir, f"global_iterations_k{int(k)}_itr{int(itr)}_recall.png")

    try:
        plot_time(df, k, itr, time_out)
        plot_recall(df, k, itr, recall_out)
    except Exception as e:
        print("Error while plotting:", e)
        sys.exit(2)

    print("Plots saved:", time_out, recall_out)


if __name__ == "__main__":
    main()
