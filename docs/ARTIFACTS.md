# Artifact Guide

## Canonical Research Data

`data/Dataset/` contains the complete saved run. Each category directory contains:

- `prompts.txt`: base and variation prompts.
- `*.png`: generated images.
- `embeddings.pt`: normalized CLIP text and image embeddings.
- `similarity_report.pt`: structured analysis output.
- `similarity_report.csv`: readable similarity table.
- `prompt_word_changes.csv`: readable word-change table.
- `similarity_delta_chart.png`: category visualization.

The dataset root also contains:

- `word_impact_summary.csv`
- `word_impact_biggest_changes.png`
- `word_impact_smallest_changes.png`

## Easy-to-Browse Copies

`results/tables/` contains CSV copies of the similarity and word-change tables without requiring users to open the larger category folders.

`examples/` contains selected images and graphs used in the README and example guide.

The canonical dataset remains under `data/Dataset`; the copies are for presentation and browsing.
