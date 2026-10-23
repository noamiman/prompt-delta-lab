from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


@dataclass(frozen=True)
class WordChangeRecord:
    """Store the word-level diff for one variation prompt."""

    category: str
    label: str
    base_prompt: str
    variation_prompt: str
    delta_similarity: float
    added_words: List[str]
    removed_words: List[str]
    replaced_words: List[str]


def tokenize_prompt(prompt: str) -> List[str]:
    """Split a prompt into normalized word tokens."""
    return [token.lower() for token in WORD_PATTERN.findall(prompt)]


def extract_prompt_changes(base_prompt: str, variation_prompt: str) -> Dict[str, List[str]]:
    """Extract added, removed, and replaced words between two prompts."""
    base_tokens = tokenize_prompt(base_prompt)
    variation_tokens = tokenize_prompt(variation_prompt)
    matcher = SequenceMatcher(a=base_tokens, b=variation_tokens)

    added_words: List[str] = []
    removed_words: List[str] = []
    replaced_words: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        removed_segment = base_tokens[i1:i2]
        added_segment = variation_tokens[j1:j2]

        if tag == "delete":
            removed_words.extend(removed_segment)
        elif tag == "insert":
            added_words.extend(added_segment)
        elif tag == "replace":
            removed_words.extend(removed_segment)
            added_words.extend(added_segment)
            replaced_words.extend(
                [f"{old}->{new}" for old, new in zip(removed_segment, added_segment)]
            )

            if len(removed_segment) > len(added_segment):
                removed_words.extend(removed_segment[len(added_segment):])
            elif len(added_segment) > len(removed_segment):
                added_words.extend(added_segment[len(removed_segment):])

    return {
        "added_words": added_words,
        "removed_words": removed_words,
        "replaced_words": replaced_words,
    }


def load_category_report(category_dir: Path) -> Dict[str, Any]:
    """Load the similarity report for one category."""
    report_path = category_dir / "similarity_report.csv"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing similarity report: {report_path}")

    with report_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty similarity report: {report_path}")

    base_rows = [row for row in rows if row["label"] == "base"]
    if not base_rows:
        raise KeyError(f"Missing base row in {report_path}")

    base_row = base_rows[0]
    variations = [row for row in rows if row["label"] != "base"]

    return {
        "base_prompt": base_row["prompt"],
        "variations": variations,
    }


def analyze_category_word_changes(category_dir: Path) -> List[WordChangeRecord]:
    """Analyze all variation prompts for one category."""
    category_report = load_category_report(category_dir)
    base_prompt = category_report["base_prompt"]
    records: List[WordChangeRecord] = []

    for row in category_report["variations"]:
        prompt_changes = extract_prompt_changes(base_prompt, row["prompt"])
        records.append(
            WordChangeRecord(
                category=category_dir.name,
                label=row["label"],
                base_prompt=base_prompt,
                variation_prompt=row["prompt"],
                delta_similarity=float(row["delta_similarity"]),
                added_words=prompt_changes["added_words"],
                removed_words=prompt_changes["removed_words"],
                replaced_words=prompt_changes["replaced_words"],
            )
        )

    return records


def write_category_word_changes(category_dir: Path, records: Sequence[WordChangeRecord]) -> Path:
    """Write the per-variation word change report for one category."""
    output_path = category_dir / "prompt_word_changes.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "category",
                "label",
                "base_prompt",
                "variation_prompt",
                "delta_similarity",
                "added_words",
                "removed_words",
                "replaced_words",
            ],
        )
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "category": record.category,
                    "label": record.label,
                    "base_prompt": record.base_prompt,
                    "variation_prompt": record.variation_prompt,
                    "delta_similarity": f"{record.delta_similarity:.3f}",
                    "added_words": " | ".join(record.added_words),
                    "removed_words": " | ".join(record.removed_words),
                    "replaced_words": " | ".join(record.replaced_words),
                }
            )

    return output_path


def aggregate_word_impacts(all_records: Sequence[WordChangeRecord]) -> List[Dict[str, Any]]:
    """Aggregate per-word impact statistics across all variation prompts."""
    stats = defaultdict(
        lambda: {
            "count": 0,
            "sum_delta": 0.0,
            "sum_abs_delta": 0.0,
            "min_delta": None,
            "max_delta": None,
        }
    )

    for record in all_records:
        changed_words = set(record.added_words + record.removed_words)
        changed_words.update(part.split("->", 1)[1] for part in record.replaced_words if "->" in part)
        changed_words.update(part.split("->", 1)[0] for part in record.replaced_words if "->" in part)

        for word in changed_words:
            entry = stats[word]
            entry["count"] += 1
            entry["sum_delta"] += record.delta_similarity
            entry["sum_abs_delta"] += abs(record.delta_similarity)
            entry["min_delta"] = record.delta_similarity if entry["min_delta"] is None else min(entry["min_delta"], record.delta_similarity)
            entry["max_delta"] = record.delta_similarity if entry["max_delta"] is None else max(entry["max_delta"], record.delta_similarity)

    summary_rows: List[Dict[str, Any]] = []
    for word, entry in stats.items():
        summary_rows.append(
            {
                "word": word,
                "count": entry["count"],
                "mean_delta_similarity": entry["sum_delta"] / entry["count"],
                "mean_abs_delta_similarity": entry["sum_abs_delta"] / entry["count"],
                "min_delta_similarity": entry["min_delta"],
                "max_delta_similarity": entry["max_delta"],
            }
        )

    summary_rows.sort(key=lambda row: (-row["mean_abs_delta_similarity"], -row["count"], row["word"]))
    return summary_rows


def write_word_summary(dataset_root: Path, summary_rows: Sequence[Dict[str, Any]]) -> Path:
    """Write the aggregated word impact summary at the dataset root."""
    output_path = dataset_root / "word_impact_summary.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "word",
                "count",
                "mean_delta_similarity",
                "mean_abs_delta_similarity",
                "min_delta_similarity",
                "max_delta_similarity",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {
                **row,
                "mean_delta_similarity": f"{row['mean_delta_similarity']:.3f}",
                "mean_abs_delta_similarity": f"{row['mean_abs_delta_similarity']:.3f}",
                "min_delta_similarity": f"{row['min_delta_similarity']:.3f}",
                "max_delta_similarity": f"{row['max_delta_similarity']:.3f}",
            }
            for row in summary_rows
        )

    return output_path


def analyze_dataset_word_changes(dataset_root: Path) -> None:
    """Analyze prompt word changes across all category reports."""
    all_records: List[WordChangeRecord] = []

    for category_dir in sorted(dataset_root.iterdir()):
        if not category_dir.is_dir():
            continue

        report_path = category_dir / "similarity_report.csv"
        if not report_path.is_file():
            continue

        print(f"Analyzing prompt changes: {category_dir.name}")
        records = analyze_category_word_changes(category_dir)
        all_records.extend(records)

        output_path = write_category_word_changes(category_dir, records)
        print(f"Saved word changes to {output_path}")

    summary_rows = aggregate_word_impacts(all_records)
    summary_path = write_word_summary(dataset_root, summary_rows)
    print(f"Saved aggregated word summary to {summary_path}")

    if summary_rows:
        biggest = summary_rows[:10]
        slightest = list(reversed(summary_rows[-10:]))

        print("Words with the biggest average change:")
        for row in biggest:
            print(
                f"  {row['word']}: mean_abs_delta={row['mean_abs_delta_similarity']:.3f}, "
                f"count={int(row['count'])}"
            )

        print("Words with the slightest average change:")
        for row in slightest:
            print(
                f"  {row['word']}: mean_abs_delta={row['mean_abs_delta_similarity']:.3f}, "
                f"count={int(row['count'])}"
            )


def main() -> None:
    """Run the prompt word change analysis across the dataset."""
    dataset_root = Path("data/Dataset")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    analyze_dataset_word_changes(dataset_root)


if __name__ == "__main__":
    main()