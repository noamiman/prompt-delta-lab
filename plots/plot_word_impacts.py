from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def load_word_summary(dataset_root: Path) -> List[Dict[str, str]]:
    """Load the aggregated word summary CSV."""
    summary_path = dataset_root / "word_impact_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing word summary file: {summary_path}")

    with summary_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        return list(reader)


def parse_summary_row(row: Dict[str, str]) -> Dict[str, float | str]:
    """Convert a CSV summary row into numeric values."""
    return {
        "word": row["word"],
        "count": int(float(row["count"])),
        "mean_delta_similarity": float(row["mean_delta_similarity"]),
        "mean_abs_delta_similarity": float(row["mean_abs_delta_similarity"]),
        "min_delta_similarity": float(row["min_delta_similarity"]),
        "max_delta_similarity": float(row["max_delta_similarity"]),
    }


def select_top_and_bottom_words(summary_rows: List[Dict[str, str]], limit: int = 10) -> tuple[List[Dict[str, float | str]], List[Dict[str, float | str]]]:
    """Select the strongest and weakest average word impacts."""
    parsed_rows = [parse_summary_row(row) for row in summary_rows]
    parsed_rows.sort(key=lambda row: (-row["mean_abs_delta_similarity"], -row["count"], row["word"]))

    biggest = parsed_rows[:limit]

    smallest_pool = sorted(
        parsed_rows,
        key=lambda row: (row["mean_abs_delta_similarity"], -row["count"], row["word"]),
    )
    smallest = smallest_pool[:limit]

    return biggest, smallest


def wrap_word_label(row: Dict[str, float | str]) -> str:
    """Create a compact display label for one word row."""
    return f"{row['word']}\n(n={int(row['count'])})"


def plot_ranked_words(
    rows: List[Dict[str, float | str]],
    title: str,
    subtitle: str,
    output_path: Path,
    reverse: bool = False,
) -> None:
    """Plot a horizontal bar chart for a ranked word list."""
    ordered_rows = list(rows)
    if reverse:
        ordered_rows = list(reversed(ordered_rows))

    labels = [wrap_word_label(row) for row in ordered_rows]
    values = [float(row["mean_abs_delta_similarity"]) for row in ordered_rows]
    signed_values = [float(row["mean_delta_similarity"]) for row in ordered_rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig_height = max(6.4, 0.55 * len(ordered_rows) + 2.0)
    fig, ax = plt.subplots(figsize=(13.8, fig_height), dpi=220)

    colors = ["#d94f41" if value >= 0 else "#1f9d8a" for value in signed_values]
    bars = ax.barh(labels, values, color=colors, edgecolor="#1f2937", linewidth=0.7, height=0.62)

    ax.set_title(title, fontsize=18, fontweight="bold", pad=18)
    ax.text(
        0.5,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#4b5563",
    )
    ax.set_xlabel("Mean absolute delta similarity", fontsize=12, labelpad=10)
    ax.set_ylabel("Changed word", fontsize=12, labelpad=10)

    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.grid(axis="x", linestyle="--", alpha=0.18)
    ax.grid(axis="y", visible=False)

    max_value = max(values) if values else 0.0
    padding = max(0.01, max_value * 0.18)
    ax.set_xlim(0.0, max_value + padding)

    for bar, row, value in zip(bars, ordered_rows, values):
        signed_value = float(row["mean_delta_similarity"])
        ax.text(
            bar.get_width() + padding * 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}  |  mean {signed_value:+.3f}",
            va="center",
            ha="left",
            fontsize=9.2,
            color="#111827",
            fontweight="semibold",
        )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfc")
    fig.subplots_adjust(left=0.25, right=0.985, top=0.90, bottom=0.08)
    fig.savefig(output_path)
    plt.close(fig)


def generate_word_impact_charts(dataset_root: Path) -> None:
    """Generate the biggest and slightest changed word charts."""
    summary_rows = load_word_summary(dataset_root)
    if not summary_rows:
        raise ValueError(f"Empty word summary: {dataset_root / 'word_impact_summary.csv'}")

    biggest, smallest = select_top_and_bottom_words(summary_rows, limit=10)

    biggest_output = dataset_root / "word_impact_biggest_changes.png"
    smallest_output = dataset_root / "word_impact_smallest_changes.png"

    plot_ranked_words(
        biggest,
        title="Words With the Biggest Prompt Impact",
        subtitle="Ranked by mean absolute change in CLIP text-image similarity",
        output_path=biggest_output,
        reverse=False,
    )
    print(f"Saved chart to {biggest_output}")

    plot_ranked_words(
        smallest,
        title="Words With the Smallest Prompt Impact",
        subtitle="Ranked by mean absolute change in CLIP text-image similarity",
        output_path=smallest_output,
        reverse=True,
    )
    print(f"Saved chart to {smallest_output}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Render charts for the strongest and weakest prompt word impacts.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Dataset"), help="Root dataset directory")
    return parser.parse_args()


def main() -> None:
    """Render the word impact charts."""
    args = parse_args()
    dataset_root = args.dataset_root

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    generate_word_impact_charts(dataset_root)


if __name__ == "__main__":
    main()