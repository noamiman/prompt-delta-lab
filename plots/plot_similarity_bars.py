from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import List

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


@dataclass(frozen=True)
class VariationScore:
    """Container for one variation row from the similarity report."""

    label: str
    prompt: str
    delta_similarity: float
    variation_similarity: float
    base_similarity: float


def load_category_scores(category_dir: Path) -> tuple[float, str, List[VariationScore]]:
    """Load the base score and all variation scores from a category CSV report."""
    report_path = category_dir / "similarity_report.csv"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing report file: {report_path}")

    base_similarity = 0.0
    base_prompt = ""
    variations: List[VariationScore] = []

    with report_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            label = row["label"].strip()
            prompt = row["prompt"].strip()
            base_value = float(row["base_similarity"])
            variation_value = float(row["variation_similarity"])
            delta_value = float(row["delta_similarity"])

            if label == "base":
                base_similarity = base_value
                base_prompt = prompt
                continue

            variations.append(
                VariationScore(
                    label=label,
                    prompt=prompt,
                    delta_similarity=delta_value,
                    variation_similarity=variation_value,
                    base_similarity=base_value,
                )
            )

    return base_similarity, base_prompt, variations


def wrap_label(label: str, prompt: str) -> str:
    """Build a compact readable label for the chart axis."""
    return label


def wrap_prompt(prompt: str) -> str:
    """Wrap prompt text for compact annotations inside the plot."""
    return fill(prompt, width=58)


def plot_category_bar_chart(category_dir: Path) -> Path:
    """Create a polished two-panel horizontal bar chart for one category."""
    base_similarity, base_prompt, variations = load_category_scores(category_dir)

    if not variations:
        raise ValueError(f"No variation rows found in {category_dir / 'similarity_report.csv'}")

    variations = sorted(variations, key=lambda item: item.delta_similarity)

    labels = [wrap_label(item.label, item.prompt) for item in variations]
    prompts = [wrap_prompt(item.prompt) for item in variations]
    deltas = [item.delta_similarity for item in variations]
    variation_similarities = [item.variation_similarity for item in variations]
    y_positions = list(range(len(variations)))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig_height = max(7.8, 0.92 * len(variations) + 2.6)
    fig, (ax_top, ax_bottom) = plt.subplots(
        nrows=2,
        ncols=1,
        sharey=True,
        figsize=(15.8, fig_height),
        dpi=220,
        gridspec_kw={"height_ratios": [1.0, 1.06], "hspace": 0.24},
    )

    positive_color = "#1f9d8a"
    negative_color = "#d94f41"
    variation_color = "#5b6ef5"
    delta_colors = [positive_color if value >= 0 else negative_color for value in deltas]

    variation_bars = ax_top.barh(
        y_positions,
        variation_similarities,
        color=variation_color,
        edgecolor="#1f2937",
        linewidth=0.7,
        height=0.54,
        alpha=0.96,
    )
    delta_bars = ax_bottom.barh(
        y_positions,
        deltas,
        color=delta_colors,
        edgecolor="#1f2937",
        linewidth=0.7,
        height=0.54,
        alpha=0.96,
    )

    ax_top.axvline(base_similarity, color="#111827", linewidth=1.15, linestyle="--", alpha=0.85)
    ax_bottom.axvline(0.0, color="#111827", linewidth=1.15, alpha=0.85)

    ax_bottom.set_xlabel("Change in similarity relative to the base pair", fontsize=12, labelpad=14)
    ax_bottom.set_yticks(y_positions)
    ax_bottom.set_yticklabels(labels)

    title = f"{category_dir.name.replace('_', ' ').title()}"
    subtitle = f"Base similarity: {base_similarity:.3f} | Base prompt: {fill(base_prompt, width=95)}"
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.982)
    fig.text(0.5, 0.944, subtitle, ha="center", va="top", fontsize=10.5, color="#374151")

    ax_top.tick_params(axis="y", labelleft=False, left=False)
    ax_bottom.tick_params(axis="y", labelsize=10, pad=10)
    ax_top.tick_params(axis="x", labelsize=10)
    ax_bottom.tick_params(axis="x", labelsize=10)
    ax_top.xaxis.set_major_locator(MaxNLocator(6))
    ax_bottom.xaxis.set_major_locator(MaxNLocator(6))

    ax_top.grid(axis="x", linestyle="--", alpha=0.18, linewidth=0.9)
    ax_bottom.grid(axis="x", linestyle="--", alpha=0.18, linewidth=0.9)
    ax_top.grid(axis="y", visible=False)
    ax_bottom.grid(axis="y", visible=False)

    variation_min = min(variation_similarities)
    variation_max = max(variation_similarities)
    variation_padding = max(0.04, (variation_max - variation_min) * 0.18)
    ax_top.set_xlim(
        max(0.0, variation_min - variation_padding),
        min(1.0, variation_max + variation_padding),
    )
    ax_top.margins(x=0.06)

    delta_min = min(deltas)
    delta_max = max(deltas)
    delta_padding = max(0.03, (delta_max - delta_min) * 0.22)
    ax_bottom.set_xlim(delta_min - delta_padding, delta_max + delta_padding)
    ax_bottom.invert_yaxis()
    ax_bottom.margins(x=0.08)

    for bar, value in zip(variation_bars, variation_similarities):
        ax_top.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=9.6,
            fontweight="bold",
            color="#111827",
        )

    for bar, value in zip(delta_bars, deltas):
        offset = 0.004 if value >= 0 else -0.004
        alignment = "left" if value >= 0 else "right"
        ax_bottom.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            va="center",
            ha=alignment,
            fontsize=9.6,
            fontweight="bold",
            color="#111827",
        )


    ax_top.text(
        0.99,
        0.10,
        f"Base line: {base_similarity:.3f}",
        transform=ax_top.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#ffffff", "edgecolor": "#d1d5db"},
    )

    ax_bottom.set_title("Delta similarity relative to base", fontsize=13, fontweight="semibold", pad=12)

    fig.patch.set_facecolor("white")
    ax_top.set_facecolor("#fbfbfc")
    ax_bottom.set_facecolor("#fbfbfc")
    fig.subplots_adjust(left=0.12, right=0.985, top=0.90, bottom=0.09, hspace=0.28)

    output_path = category_dir / "similarity_delta_chart.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_dataset_charts(dataset_root: Path) -> None:
    """Generate a bar chart for every category that has a similarity report."""
    for category_dir in sorted(dataset_root.iterdir()):
        if not category_dir.is_dir():
            continue

        report_path = category_dir / "similarity_report.csv"
        if not report_path.is_file():
            continue

        print(f"Rendering chart for category: {category_dir.name}")
        output_path = plot_category_bar_chart(category_dir)
        print(f"Saved chart to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Render CLIP similarity bar charts for each dataset category.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Dataset"), help="Root dataset directory")
    return parser.parse_args()


def main() -> None:
    """Render bar charts for all category similarity reports."""
    args = parse_args()
    dataset_root = args.dataset_root

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    generate_dataset_charts(dataset_root)


if __name__ == "__main__":
    main()