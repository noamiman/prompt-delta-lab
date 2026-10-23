from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any

import torch


def compute_cosine_similarity(text_embedding: torch.Tensor, image_embedding: torch.Tensor) -> float:
    """Return the cosine similarity for two L2-normalized embeddings."""
    return float(torch.dot(text_embedding, image_embedding).item())


def compute_pair_delta(
    base_text_embedding: torch.Tensor,
    base_image_embedding: torch.Tensor,
    variation_text_embedding: torch.Tensor,
    variation_image_embedding: torch.Tensor,
) -> Dict[str, float]:
    """Compute base and variation alignment scores for a prompt-image pair."""
    base_similarity = compute_cosine_similarity(base_text_embedding, base_image_embedding)
    variation_similarity = compute_cosine_similarity(variation_text_embedding, variation_image_embedding)
    text_delta_similarity = compute_cosine_similarity(base_text_embedding, variation_text_embedding)
    image_delta_similarity = compute_cosine_similarity(base_image_embedding, variation_image_embedding)

    return {
        "base_similarity": base_similarity,
        "variation_similarity": variation_similarity,
        "delta_similarity": variation_similarity - base_similarity,
        "text_delta_similarity": text_delta_similarity,
        "image_delta_similarity": image_delta_similarity,
    }


def load_category_embeddings(category_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load the saved embeddings dictionary for a single category."""
    embeddings_path = category_dir / "embeddings.pt"
    return torch.load(embeddings_path, map_location="cpu", weights_only=False)


def analyze_category(category_dir: Path) -> Dict[str, Any]:
    """Analyze one category folder and produce per-variation similarity deltas."""
    embeddings = load_category_embeddings(category_dir)

    if "base" not in embeddings:
        raise KeyError(f"Missing base entry in {category_dir / 'embeddings.pt'}")

    base_entry = embeddings["base"]
    base_text_embedding = base_entry["text_embedding"]
    base_image_embedding = base_entry["image_embedding"]

    category_report: Dict[str, Any] = {
        "base": {
            "prompt": base_entry["prompt"],
            "image_path": base_entry["image_path"],
            "base_similarity": compute_cosine_similarity(base_text_embedding, base_image_embedding),
        },
        "variations": {},
    }

    for label, entry in embeddings.items():
        if label == "base":
            continue

        scores = compute_pair_delta(
            base_text_embedding=base_text_embedding,
            base_image_embedding=base_image_embedding,
            variation_text_embedding=entry["text_embedding"],
            variation_image_embedding=entry["image_embedding"],
        )

        category_report["variations"][label] = {
            "prompt": entry["prompt"],
            "image_path": entry["image_path"],
            **scores,
        }

    return category_report


def write_category_csv(category_dir: Path, category_report: Dict[str, Any]) -> Path:
    """Write a flat CSV summary for one analyzed category."""
    output_path = category_dir / "similarity_report.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "label",
                "prompt",
                "image_path",
                "base_similarity",
                "variation_similarity",
                "delta_similarity",
                "text_delta_similarity",
                "image_delta_similarity",
            ],
        )
        writer.writeheader()

        base_entry = category_report["base"]
        writer.writerow(
            {
                "label": "base",
                "prompt": base_entry["prompt"],
                "image_path": base_entry["image_path"],
                "base_similarity": f"{base_entry['base_similarity']:.3f}",
                "variation_similarity": f"{base_entry['base_similarity']:.3f}",
                "delta_similarity": "0.000",
                "text_delta_similarity": "1.000",
                "image_delta_similarity": "1.000",
            }
        )

        for label, entry in category_report["variations"].items():
            writer.writerow(
                {
                    "label": label,
                    "prompt": entry["prompt"],
                    "image_path": entry["image_path"],
                    "base_similarity": f"{entry['base_similarity']:.3f}",
                    "variation_similarity": f"{entry['variation_similarity']:.3f}",
                    "delta_similarity": f"{entry['delta_similarity']:.3f}",
                    "text_delta_similarity": f"{entry['text_delta_similarity']:.3f}",
                    "image_delta_similarity": f"{entry['image_delta_similarity']:.3f}",
                }
            )

    return output_path


def analyze_dataset(dataset_root: Path) -> None:
    """Analyze all category folders that contain saved embeddings."""
    for category_dir in sorted(dataset_root.iterdir()):
        if not category_dir.is_dir():
            continue

        embeddings_path = category_dir / "embeddings.pt"
        if not embeddings_path.is_file():
            continue

        print(f"Analyzing category: {category_dir.name}")
        category_report = analyze_category(category_dir)

        output_path = category_dir / "similarity_report.pt"
        torch.save(category_report, output_path)
        print(f"Saved analysis to {output_path}")

        csv_path = write_category_csv(category_dir, category_report)
        print(f"Saved CSV summary to {csv_path}")

        base_similarity = category_report["base"]["base_similarity"]
        for label, entry in category_report["variations"].items():
            print(
                f"  {label}: delta={entry['delta_similarity']:.3f}, "
                f"variation={entry['variation_similarity']:.3f}, base={base_similarity:.3f}"
            )


def main() -> None:
    """Run the CLIP similarity analysis across the dataset."""
    dataset_root = Path("data/Dataset")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    analyze_dataset(dataset_root)


if __name__ == "__main__":
    main()