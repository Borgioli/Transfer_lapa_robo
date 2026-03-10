# When Self-Supervision Transfers: Quantifying the Laparoscopic–Robotic Domain Shift

This is the anonymized companion repository for the IROS 2026 submission *"When Self-Supervision Transfers: Quantifying the Laparoscopic–Robotic Domain Shift"*.

## Contents

This repository contains:

- **Trained models** — All model checkpoints used for evaluation in the paper (EndoViT backbone + SPR phase-classification head under frozen, partially unfrozen, and fully unfrozen regimes).
- **Reviewer main scripts** — `reviewer_main_scripts/` contains a curated copy of the main pretraining, fine-tuning, evaluation, and sweep entrypoints pulled from the local `endovit/` and `endovit_vanilla/` codebases, plus a checkpoint inventory.
- **Testing scripts** — Scripts to reproduce the four experiments reported in the paper:
  - **E1** – Cross-domain evaluation (laparoscopic ↔ robotic transfer).
  - **E2** – Frozen / partially unfrozen / fully unfrozen fine-tuning ablation.
  - **E3** – Progressive laparoscopic data injection.
  - **E4** – Robotic-only vs. mixed MAE pretraining comparison.
- **Figure generation scripts** — Python scripts to reproduce the figures and tables in the paper:
  - `gen_e4_tables_hist.py` — Generates the E4 grouped bar chart (Table VII / VIII comparison).
  - `gen_injection_fig.py` — Generates the E3 injection curve figure.
  - `make_injection_fig.py` / `make_injection_fig_v2.py` — Alternative injection figure variants with class-distribution plots.
- **Presentation generation scripts** — Export the summary deck in multiple formats:
  - `generate_pdf_slides.py` — Generates the static PDF slide deck.
  - `generate_ppt_slides.py` — Generates an editable `.pptx` deck with native PowerPoint text, tables, and charts.

## Requirements

```
numpy
matplotlib
```

## Citation

If you find this work useful, please cite:

```
[Citation will be added upon acceptance]
```

## License

This repository is provided for anonymous review purposes only.
