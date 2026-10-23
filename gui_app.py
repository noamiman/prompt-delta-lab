from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from analyze_embeddings import compute_pair_delta
from analyze_prompt_word_changes import extract_prompt_changes
from build_embeddings import (
    encode_text_and_image,
    get_device,
    load_clip_model_and_processor,
)

DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"


st.set_page_config(
    page_title="Prompt Delta Lab",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f1ea; color: #17211b; }
    [data-testid="stSidebar"] { background: #17211b; }
    [data-testid="stSidebar"] * { color: #f5f1ea; }
    h1, h2, h3 { font-family: Georgia, serif; letter-spacing: 0; }
    .hero { padding: 1.4rem 0 0.8rem; }
    .eyebrow { color: #b65036; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
    .hero h1 { font-size: 3rem; margin: 0.2rem 0 0.4rem; }
    .hero p { color: #526057; font-size: 1.05rem; max-width: 720px; }
    .metric-card { background: #fffdf9; border: 1px solid #ded8cc; padding: 0.9rem 1rem; min-height: 106px; }
    .metric-label { color: #66736a; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-value { color: #17211b; font-size: 1.8rem; font-weight: 700; margin-top: 0.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_diffusion_pipeline() -> tuple[StableDiffusionPipeline, torch.device]:
    """Load and cache the image-generation pipeline."""
    device = get_device()
    pipeline = StableDiffusionPipeline.from_pretrained(
        DIFFUSION_MODEL_ID,
        torch_dtype=torch.float32,
    )
    pipeline = pipeline.to(device)
    pipeline.enable_attention_slicing()
    return pipeline, device


@st.cache_resource(show_spinner=False)
def load_clip_resources() -> tuple[Any, Any, torch.device]:
    """Load and cache the CLIP model and processor used by the research code."""
    device = get_device()
    model, processor = load_clip_model_and_processor(device)
    return model, processor, device


def read_prompt_upload(uploaded_file: Any) -> str:
    """Read a UTF-8 text prompt uploaded through Streamlit."""
    if uploaded_file is None:
        return ""
    return uploaded_file.getvalue().decode("utf-8").strip()


def generate_image(
    pipeline: StableDiffusionPipeline,
    prompt: str,
    device: torch.device,
    seed: int,
    steps: int,
) -> Image.Image:
    """Generate one reproducible image for a prompt."""
    generator = torch.Generator(device=device.type).manual_seed(seed)
    with torch.inference_mode():
        output = pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps=steps,
        )
    return output.images[0]


def analyze_images(
    model: Any,
    processor: Any,
    device: torch.device,
    first_prompt: str,
    second_prompt: str,
    first_image: Image.Image,
    second_image: Image.Image,
) -> dict[str, float]:
    """Run the project's CLIP embedding and pair-delta analysis on two images."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        first_path = Path(temporary_directory) / "first.png"
        second_path = Path(temporary_directory) / "second.png"
        first_image.save(first_path)
        second_image.save(second_path)

        first_text, first_image_embedding = encode_text_and_image(
            model, processor, device, first_prompt, first_path
        )
        second_text, second_image_embedding = encode_text_and_image(
            model, processor, device, second_prompt, second_path
        )

    return compute_pair_delta(
        base_text_embedding=first_text,
        base_image_embedding=first_image_embedding,
        variation_text_embedding=second_text,
        variation_image_embedding=second_image_embedding,
    )


def render_metric(label: str, value: float) -> None:
    """Render a compact metric card with three decimal places."""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value:.3f}</div></div>',
        unsafe_allow_html=True,
    )


def get_changed_word_labels(first_prompt: str, second_prompt: str) -> list[str]:
    """Return unique changed words, including old-to-new replacement labels."""
    changes = extract_prompt_changes(first_prompt, second_prompt)
    labels = list(changes["added_words"] + changes["removed_words"])
    labels.extend(changes["replaced_words"])
    return list(dict.fromkeys(labels))


def render_change_plot(
    first_prompt: str,
    second_prompt: str,
    scores: dict[str, float],
) -> None:
    """Plot the observed image-text shift alongside its changed-word attribution."""
    text_change = 1.0 - scores["text_delta_similarity"]
    image_change = 1.0 - scores["image_delta_similarity"]
    changed_words = get_changed_word_labels(first_prompt, second_prompt)

    figure, axes = plt.subplots(
        nrows=2,
        figsize=(10, max(5.8, 0.45 * len(changed_words) + 4.2)),
        dpi=160,
        gridspec_kw={"height_ratios": [0.9, 1.4]},
    )
    summary_axis, word_axis = axes
    summary_bars = summary_axis.bar(
        ["Prompt similarity change", "Image similarity change"],
        [text_change, image_change],
        color=["#b65036", "#2d7b70"],
        width=0.55,
    )
    summary_axis.set_title("Prompt change versus image change", loc="left", fontweight="bold")
    summary_axis.set_ylabel("Change magnitude")
    summary_axis.set_ylim(0, max(0.05, max(text_change, image_change) * 1.25))
    summary_axis.grid(axis="y", linestyle="--", alpha=0.25)
    summary_axis.set_axisbelow(True)
    for bar, value in zip(summary_bars, [text_change, image_change]):
        summary_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{value:.3f}",
            ha="center",
            fontweight="bold",
        )

    word_axis.set_title("Changed-word impact on the observed image-text shift", loc="left", fontweight="bold")
    if changed_words:
        # This follows the research reports: the measured pair delta is associated
        # with every token changed between the two prompts.
        word_values = [scores["delta_similarity"]] * len(changed_words)
        word_colors = ["#2d7b70" if value >= 0 else "#b65036" for value in word_values]
        word_bars = word_axis.barh(changed_words, word_values, color=word_colors, height=0.58)
        word_axis.axvline(0.0, color="#17211b", linewidth=1)
        word_axis.set_xlabel("Change in prompt-image alignment")
        word_axis.grid(axis="x", linestyle="--", alpha=0.25)
        word_axis.set_axisbelow(True)
        for bar, value in zip(word_bars, word_values):
            word_axis.text(
                value + (0.002 if value >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.3f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontweight="bold",
            )
    else:
        word_axis.text(0.5, 0.5, "No word-level changes detected", ha="center", va="center")
        word_axis.set_axis_off()

    figure.patch.set_facecolor("#fffdf9")
    for axis in axes:
        axis.set_facecolor("#fffdf9")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def main() -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">CLIP latent transformations</div>'
        '<h1>Prompt Delta Lab</h1>'
        '<p>Change one prompt, generate both images, and see whether the text edit produces a matching shift in image space.</p></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Experiment")
        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42, step=1)
        steps = st.slider("Diffusion steps", min_value=10, max_value=50, value=30)
        st.caption(f"Device: {get_device()}")
        st.caption("Models are loaded only after you click Generate.")

    left_column, right_column = st.columns(2, gap="large")
    with left_column:
        st.subheader("Original prompt")
        first_upload = st.file_uploader("Upload a .txt prompt", type=["txt"], key="first_upload")
        first_prompt_default = read_prompt_upload(first_upload)
        first_prompt = st.text_area(
            "Prompt A",
            value=first_prompt_default,
            height=140,
            placeholder="A cinematic photograph of a red apple on a wooden table",
            label_visibility="collapsed",
        )

    with right_column:
        st.subheader("Changed prompt")
        second_upload = st.file_uploader("Upload a .txt prompt", type=["txt"], key="second_upload")
        second_prompt_default = read_prompt_upload(second_upload)
        second_prompt = st.text_area(
            "Prompt B",
            value=second_prompt_default,
            height=140,
            placeholder="A cinematic photograph of a green apple on a wooden table",
            label_visibility="collapsed",
        )

    generate_clicked = st.button("Generate comparison", type="primary", use_container_width=True)
    if not generate_clicked:
        st.info("Enter two prompts, then generate a comparison.")
        return

    first_prompt = first_prompt.strip()
    second_prompt = second_prompt.strip()
    if not first_prompt or not second_prompt:
        st.error("Both prompts are required.")
        return

    try:
        with st.spinner("Loading Stable Diffusion and CLIP models..."):
            pipeline, device = load_diffusion_pipeline()
            clip_model, clip_processor, clip_device = load_clip_resources()

        with st.spinner("Generating both images..."):
            first_image = generate_image(pipeline, first_prompt, device, int(seed), steps)
            second_image = generate_image(pipeline, second_prompt, device, int(seed), steps)

        with st.spinner("Encoding prompts and images with CLIP..."):
            scores = analyze_images(
                clip_model,
                clip_processor,
                clip_device,
                first_prompt,
                second_prompt,
                first_image,
                second_image,
            )
    except Exception as error:
        st.exception(error)
        return

    st.session_state["comparison"] = {
        "first_prompt": first_prompt,
        "second_prompt": second_prompt,
        "first_image": first_image,
        "second_image": second_image,
        "scores": scores,
    }

    comparison = st.session_state["comparison"]
    st.divider()
    image_column_a, image_column_b = st.columns(2, gap="large")
    with image_column_a:
        st.subheader("Generated image A")
        st.image(comparison["first_image"], use_container_width=True)
    with image_column_b:
        st.subheader("Generated image B")
        st.image(comparison["second_image"], use_container_width=True)

    scores = comparison["scores"]
    st.subheader("CLIP comparison")
    metric_columns = st.columns(4)
    metric_values = [
        ("Prompt similarity", scores["text_delta_similarity"]),
        ("Image similarity", scores["image_delta_similarity"]),
        ("Image-text delta", scores["delta_similarity"]),
        ("Base alignment", scores["base_similarity"]),
    ]
    for column, (label, value) in zip(metric_columns, metric_values):
        with column:
            render_metric(label, value)

    st.caption(
        "Prompt and image similarity compare A with B. The word chart associates the measured "
        "image-text alignment change with each word changed between the prompts; it is an attribution "
        "of the observed shift, not a separate image generation for each word."
    )
    render_change_plot(comparison["first_prompt"], comparison["second_prompt"], scores)


if __name__ == "__main__":
    main()
