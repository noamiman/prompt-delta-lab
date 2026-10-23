from pathlib import Path
import re
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

import dataset_prompts

def generate_image_with_fixed_seed(
    prompt: str, 
    pipeline: StableDiffusionPipeline, 
    seed: int = 42, 
    steps: int = 30
) -> Image.Image:
    """
    Generates a single image from a text prompt using a fixed seed on Apple Silicon (MPS).
    
    Args:
        prompt (str): The text description for the image.
        pipeline (StableDiffusionPipeline): The pre-loaded Hugging Face diffusion pipeline.
        seed (int): The exact seed integer to enforce structural composition. Defaults to 42.
        steps (int): Number of inference steps for the diffusion process. Defaults to 30.
        
    Returns:
        PIL.Image.Image: The generated image object.
    """
    # 1. Enforce strict seed consistency by resetting the MPS generator right before generation
    generator = torch.Generator(device=pipeline.device.type).manual_seed(seed)
    
    # 2. Run the inference pipeline
    output = pipeline(
        prompt=prompt,
        generator=generator,
        num_inference_steps=steps
    )
    
    # 3. Extract and return the single PIL image
    return output.images[0]


def sanitize_name(text: str) -> str:
    """Convert prompt text into a safe file name fragment."""
    cleaned_text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")
    return cleaned_text[:80] or "prompt"


def save_prompt_image(
    image: Image.Image,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def generate_dataset_images(
    pipeline: StableDiffusionPipeline,
    dataset: dict,
    output_root: Path,
    seed: int = 42,
    steps: int = 30,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    for category_name, content in dataset.items():
        category_dir = output_root / category_name
        category_dir.mkdir(parents=True, exist_ok=True)

        prompt_groups = [("base", content["base"])]
        prompt_groups.extend(
            (f"variation_{index:02d}", variation)
            for index, variation in enumerate(content["variations"], start=1)
        )

        prompts_file = category_dir / "prompts.txt"
        with prompts_file.open("w", encoding="utf-8") as file_handle:
            file_handle.write(f"base: {content['base']}\n")
            for index, variation in enumerate(content["variations"], start=1):
                file_handle.write(f"variation_{index:02d}: {variation}\n")

        for prompt_label, prompt_text in prompt_groups:
            print(f"Generating {category_name}/{prompt_label}: {prompt_text}")
            generated_image = generate_image_with_fixed_seed(
                prompt=prompt_text,
                pipeline=pipeline,
                seed=seed,
                steps=steps,
            )
            output_path = category_dir / f"{prompt_label}_{sanitize_name(prompt_text)}.png"
            save_prompt_image(generated_image, output_path)


# --- Quick Usage Example ---
# model: runwayml/stable-diffusion-v1-5
if __name__ == "__main__":
    print("Loading pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
    pipe = pipe.to("mps")
    pipe.enable_attention_slicing()

    generate_dataset_images(
        pipeline=pipe,
        dataset=dataset_prompts.PROMPT_DATASET,
        output_root=Path("data/Dataset"),
        seed=42,
        steps=30,
    )

    print("Saved dataset images successfully.")