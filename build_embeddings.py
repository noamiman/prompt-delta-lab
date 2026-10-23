from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def get_device() -> torch.device:
    """Return the preferred inference device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# openai/clip-vit-large-patch14 is the largest CLIP model available in the Hugging Face Transformers library.
def load_clip_model_and_processor(device: torch.device) -> Tuple[CLIPModel, CLIPProcessor]:
    """Load the official OpenAI CLIP checkpoint and move the model to the target device."""
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.to(device)
    model.eval()
    return model, processor


def parse_prompts_file(prompts_path: Path) -> Dict[str, str]:
    """Parse a prompts.txt file into a label-to-prompt mapping."""
    prompts: Dict[str, str] = {}

    with prompts_path.open("r", encoding="utf-8") as file_handle:
        for raw_line in file_handle:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue

            label, prompt_text = line.split(":", 1)
            label = label.strip()
            prompt_text = prompt_text.strip()

            if label and prompt_text:
                prompts[label] = prompt_text

    return prompts

def find_image_for_label(category_dir: Path, label: str) -> Path | None:
    """Find the PNG image that corresponds to a prompt label."""
    matches = sorted(category_dir.glob(f"{label}_*.png"))
    if not matches:
        return None

    return matches[0]


def encode_text_and_image(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    prompt: str,
    image_path: Path,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode and L2-normalize the text and image representations for a single item."""
    image = Image.open(image_path).convert("RGB")

    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True, truncation=True)
    image_inputs = processor(images=image, return_tensors="pt")

    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    image_inputs = {key: value.to(device) for key, value in image_inputs.items()}

    with torch.inference_mode():
        text_outputs = model.text_model(**text_inputs, return_dict=True)
        image_outputs = model.vision_model(**image_inputs, return_dict=True)

        text_pooler_output = text_outputs.pooler_output
        image_pooler_output = image_outputs.pooler_output

        text_features = model.text_projection(text_pooler_output)
        image_features = model.visual_projection(image_pooler_output)

    text_features = F.normalize(text_features, p=2, dim=-1).squeeze(0).cpu()
    image_features = F.normalize(image_features, p=2, dim=-1).squeeze(0).cpu()

    return text_features, image_features


def process_category(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    category_dir: Path,
) -> Dict[str, Dict[str, torch.Tensor | str]]:
    """Build the embeddings dictionary for a single category folder."""
    prompts_path = category_dir / "prompts.txt"
    prompts = parse_prompts_file(prompts_path)

    category_items: Dict[str, Dict[str, torch.Tensor | str]] = {}

    for label, prompt in prompts.items():
        image_path = find_image_for_label(category_dir, label)
        if image_path is None:
            print(f"Skipping {category_dir.name}/{label}: no matching PNG image found.")
            continue

        text_embedding, image_embedding = encode_text_and_image(
            model=model,
            processor=processor,
            device=device,
            prompt=prompt,
            image_path=image_path,
        )

        category_items[label] = {
            "prompt": prompt,
            "image_path": str(image_path),
            "text_embedding": text_embedding,
            "image_embedding": image_embedding,
        }

        print(f"Processed {category_dir.name}/{label}")

    return category_items


def process_dataset(dataset_root: Path) -> None:
    """Traverse the dataset root and save per-category CLIP embeddings."""
    device = get_device()
    print(f"Using device: {device}")

    model, processor = load_clip_model_and_processor(device)

    for category_dir in sorted(dataset_root.iterdir()):
        if not category_dir.is_dir():
            continue

        prompts_path = category_dir / "prompts.txt"
        if not prompts_path.is_file():
            continue

        print(f"Processing category: {category_dir.name}")
        category_embeddings = process_category(model, processor, device, category_dir)

        if not category_embeddings:
            print(f"Skipping save for {category_dir.name}: no items were processed.")
            continue

        output_path = category_dir / "embeddings.pt"
        torch.save(category_embeddings, output_path)
        print(f"Saved embeddings to {output_path}")


def main() -> None:
    """Run the dataset embedding pipeline."""
    dataset_root = Path("data/Dataset")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    process_dataset(dataset_root)


if __name__ == "__main__":
    main()