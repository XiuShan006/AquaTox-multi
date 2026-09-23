# Released multi-representation model

This directory contains five outer-fold checkpoints for the AquaTox-multi workflow. Each fold is a multi-representation model for six aquatic toxicity tasks.

Each fold includes:

- `best_model.pt` — model checkpoint;
- `operative_config.gin` — corresponding run configuration;
- `run_manifest.json` — checkpoint metadata.

Model architecture details are implemented in [`../src/rfm/models/`](../src/rfm/models/). Download the checkpoints with Git LFS.
