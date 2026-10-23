# Pipeline Overview

Prompt Delta Lab measures how controlled prompt edits relate to changes in generated images.

## Interactive Path

```text
Two prompts or .txt uploads
        |
        v
                   gui_app.py
        |              |
        v              v
 Stable Diffusion   CLIP embeddings
        |              |
        |              v
        |       similarity metrics
        |              |
        +-------> images, metrics, word-change plot
```

Launch with:

```bash
python3 -m streamlit run gui_app.py
```

The GUI uses `generate_image` for two images, `build_embeddings.py` for CLIP encoding, `analyze_embeddings.py` for pair metrics, and `analyze_prompt_word_changes.py` for changed-word extraction.

## Reproducible Research Path

```text
dataset_prompts.py
        |
        v
generate_images.py
        |
        +--> data/Dataset/<category>/prompts.txt
        +--> data/Dataset/<category>/*.png
        v
build_embeddings.py
        |
        +--> data/Dataset/<category>/embeddings.pt
        v
analyze_embeddings.py
        |
        +--> similarity_report.pt
        +--> similarity_report.csv
        v
analyze_prompt_word_changes.py
        |
        +--> prompt_word_changes.csv
        +--> word_impact_summary.csv
        v
plots/plot_*.py
        |
        +--> similarity_delta_chart.png
        +--> word_impact_*.png
```

Run the pipeline from the project root:

```bash
python3 generate_images.py
python3 build_embeddings.py
python3 analyze_embeddings.py
python3 analyze_prompt_word_changes.py
python3 -m plots.plot_similarity_bars
python3 -m plots.plot_word_impacts
```

## Artifact Locations

- `data/Dataset/`: canonical complete research run, including images, embeddings, CSV reports, and generated charts.
- `results/tables/`: easy-to-browse copies of all category CSV tables, organized by category.
- `examples/`: selected images and graphs for explaining the strongest positive and negative examples.
- `docs/`: methodology, artifact guide, and example interpretations.
