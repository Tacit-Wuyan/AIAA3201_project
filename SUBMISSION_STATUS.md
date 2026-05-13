# Submission Status Checklist

This repository is prepared for GitHub upload.

## Satisfied in this repository

- Part 1 code: `part1_pipeline.py`
- Part 2 code: `part2_pipeline.py`
- Part 3 code: `part3_pipeline.py`
- Dependencies: `requirements.txt`
- Usage instructions: `README.md`
- Mandatory input examples: `data/sample/bmx-trees.mp4`, `data/sample/tennis.mp4`, `data/wild/gymnastics_easy_long_12s_720p.mp4`
- Method flowcharts: `assets/flowchart_*.png`
- Latest Part 1 / Part 2 qualitative result figures: `assets/bmx_results.png`, `assets/tennis_results.png`, `assets/wild_gymnastics_results.png`
- Latest Part 1 / Part 2 metric summary for datasets with GT: `assets/part12_results_summary.csv`
- Part 3 adaptive refinement code: `part3_pipeline.py`
- Part 3 ablation tool: `tools/part3_ablation.py`
- Part 3 breakdance qualitative comparison: `assets/part3_breakdance_final_comparison.png`
- Part 3 ablation figure and CSV: `assets/part3_breakdance_ablation.png`, `assets/part3_breakdance_ablation_summary.csv`
- Evaluation script: `evaluate_metrics.py`

## Metric Policy Update

The course update says quantitative metrics are not required for mandatory datasets that do not provide ground truth. We therefore report JM/JR only where GT masks are available, and use qualitative comparison for no-GT mandatory videos such as the wild gymnastics video.

## Still needed outside GitHub

- `videos.zip` for Canvas / course submission, containing processed videos for wild video, bmx-trees, and tennis.
- Final PDF report in CVPR format, 6-8 pages excluding references.
- Correct citations in the report for YOLOv8, SAM2, ProPainter, DAVIS/sample data, and any Part 3 method used.
- Optional DAVIS qualitative examples have been added under `data/sample/davis_extra/`: walking, bike-packing, and breakdance. More DAVIS GT-mask experiments can still be added if aiming for a stronger quantitative section.

