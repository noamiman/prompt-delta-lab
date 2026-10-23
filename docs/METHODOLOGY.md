# Methodology

## Research Question

How strongly does a controlled change in a text prompt correspond to a change in the image generated from that prompt?

## Experimental Design

Each category contains one base prompt and five variations. A variation changes one or more semantic or stylistic elements while preserving much of the surrounding prompt. Stable Diffusion generates one image per prompt with a fixed seed and inference-step count.

The fixed seed improves comparability by reducing one source of randomness, but it does not make the images identical or make the result deterministic across every hardware and software configuration.

## Embeddings

`build_embeddings.py` uses `openai/clip-vit-large-patch14` to create:

- A normalized text embedding for each prompt.
- A normalized image embedding for each generated image.

Because the embeddings are normalized, their dot product is cosine similarity.

## Metrics

For every variation, `analyze_embeddings.py` reports:

- `text_delta_similarity`: cosine similarity between base and variation text embeddings.
- `image_delta_similarity`: cosine similarity between base and variation image embeddings.
- `base_similarity`: similarity between the base prompt and base image.
- `variation_similarity`: similarity between the variation prompt and variation image.
- `delta_similarity`: `variation_similarity - base_similarity`.

A high text or image similarity means a small movement in the relevant CLIP space. A low value means a larger movement. `delta_similarity` is different: positive values mean the variation pair is more aligned than the base pair, and negative values mean it is less aligned.

## Word-Level Analysis

`analyze_prompt_word_changes.py` uses a sequence diff to identify added, removed, and replaced prompt tokens. It joins those changes to the measured similarity delta and aggregates statistics across categories.

The word-impact tables answer:

> When a word was changed in these experiments, what alignment change was observed for the whole prompt/image pair?

They do **not** isolate a causal effect for one word. A single variation can change several words, and the generated image is produced from the complete prompt. To estimate an individual word effect more directly, run controlled one-word-at-a-time experiments while keeping all other tokens fixed and repeat across seeds.

## Interpreting Results

The measurements are useful for comparing relative behavior within this experiment. They should not be interpreted as:

- Percentages.
- Human perceptual ratings.
- A universal ranking of words.
- Proof that CLIP or Stable Diffusion understands a word independently of context.

Results depend on the prompt wording, category, model checkpoints, seed, sampler, inference steps, hardware, and preprocessing.
