# Results Manifest

This repository contains code, compact sample data, report assets, and representative result videos for Project 3: Video Object Removal and Inpainting.

## Main Code

- `part1_pipeline.py`: Part 1 classical baseline.
- `part2_pipeline.py`: Part 2 YOLO prompt + SAM2 propagation + ProPainter pipeline.
- `part3_pipeline.py`: Part 3 adaptive motion-guided refinement.
- `part23_integrated_pipeline.py`: Unified Part 2 / Part 3A runner.
- `part3_sam3_pipeline.py`: Part 3B SAM3 + ProPainter / DiffuEraser exploration.
- `evaluate_metrics.py`: JM/JR evaluation script.
- `tools/`: adapters for SAM2, SAM3, ProPainter, DiffuEraser, and setup checks.
- `configs/`: example configurations for Part 2 and Part 3 variants.

## Input Videos Included

- `data/sample/bmx-trees.mp4`
- `data/sample/tennis.mp4`
- `data/wild/gymnastics_easy_long_12s_720p.mp4`
- `data/sample/davis_extra/davis_walking.mp4`
- `data/sample/davis_extra/davis_bike-packing.mp4`
- `data/sample/davis_extra/davis_breakdance.mp4`

## Representative Processed Videos

### Part 1 / Part 2 Mandatory and Wild Results

Located in `sample_results/part12_final/`:

- `bmx_trees/part1_inpainted.mp4`
- `bmx_trees/part1_mask.mp4`
- `bmx_trees/part2_inpainted.mp4`
- `bmx_trees/part2_mask.mp4`
- `tennis/part1_inpainted.mp4`
- `tennis/part1_mask.mp4`
- `tennis/part2_inpainted.mp4`
- `tennis/part2_mask.mp4`
- `wild_gymnastics/part1_inpainted.mp4`
- `wild_gymnastics/part1_mask.mp4`
- `wild_gymnastics/part2_inpainted.mp4`
- `wild_gymnastics/part2_mask.mp4`
- `part12_results_summary.csv`

### Part 3 SAM3 / DiffuEraser Sample Results

Located in `sample_results/part3_sam3/`:

- `davis/part2_tennis_inpainted.mp4`
- `davis/part3_sam3_propainter_tennis_inpainted.mp4`
- `davis/part3_sam3_diffueraser_boat_inpainted.mp4`
- `wild/part1_main_inpainted.mp4`
- `wild/sam2_diffueraser_inpainted.mp4`
- `wild/sam3_diffueraser_inpainted.mp4`
- `wild/sam3_propainter_inpainted.mp4`

## Metrics and Report Assets

Located in `assets/`:

- `part12_results_summary.csv`: final bmx-trees / tennis Part 1 and Part 2 JM/JR metrics.
- `davis_full_part12_summary.md`: full DAVIS Part 1/2 summary.
- `davis_full_part12_part1_metrics.csv`: full DAVIS Part 1 per-sequence metrics.
- `davis_full_part12_part2_metrics.csv`: full DAVIS Part 2 per-sequence metrics.
- `davis_extra_gt_metrics.csv`: selected DAVIS stress-case metrics.
- `part3_breakdance_ablation_summary.csv`: Part 3 adaptive-refinement ablation summary.
- `part3_davis_sam3_vs_sam2_comparison.csv`: SAM3 vs SAM2 comparison on DAVIS.
- `flowchart_*.png`: report flowcharts.
- `part12_improved_final_*_comparison.png`: Part 1/2 qualitative comparisons.
- `part3_breakdance_*.png`: Part 3 ablation and final comparison figures.

## Report Files

- `Report.pdf`: final compiled report PDF.
- `report_final.tex`: CVPR-template final report source.
- `main.bib`: bibliography used by the report.
- `cvpr.sty`, `preamble.tex`, `ieeenat_fullname.bst`: LaTeX support files.
- `REPORT.md`: markdown report draft.

## Files Intentionally Excluded

- model weights (`*.pt`, `*.pth`, `*.ckpt`, etc.)
- third-party repositories under `third_party/`
- full generated frame folders and raw intermediate outputs
- local caches and backup directories

These files can be regenerated or downloaded following the README instructions.
