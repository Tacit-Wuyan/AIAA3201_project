# Video Object Removal and Inpainting

Course project for video object removal and inpainting. This repository contains three parts:

- Part 1: traditional baseline using object segmentation, motion filtering, temporal smoothing, and OpenCV inpainting.
- Part 2: SOTA-style reproduction using YOLO prompts, SAM2 video mask propagation, and ProPainter.
- Part 3: adaptive motion-aware mask refinement with target continuity and ablation analysis.

The repository is kept GitHub-friendly. Large generated outputs, model weights, caches, and third-party repositories are not committed.

## Repository Structure

```text
.
|-- part1_pipeline.py                 # Part 1 traditional baseline
|-- part2_pipeline.py                 # Part 2 SAM2 + ProPainter runner
|-- part23_integrated_pipeline.py      # Unified Part 2 / adaptive Part 3 runner
|-- part3_pipeline.py                 # Part 3 adaptive motion-refinement branch
|-- part3_sam3_pipeline.py            # Part 3 SAM3 + ProPainter / DiffuEraser branch
|-- PART3_INTEGRATION.md              # Explains how the two Part 3 branches fit together
|-- evaluate_metrics.py               # JM/JR and optional PSNR/SSIM evaluation
|-- requirements.txt                  # Base dependencies
|-- requirements_part3_sam3.txt       # Optional SAM3 / DiffuEraser branch dependencies
|-- configs/
|   |-- part2_example.yaml
|   |-- part2_motion_aware.yaml
|   |-- part2_motion_strict.yaml
|   |-- part3_adaptive_motion_refinement.yaml
|   `-- part3_sam3_*.yaml
|-- tools/
|   |-- sam2_auto_mask.py
|   |-- sam3_auto_mask.py
|   |-- propainter_adapter.py
|   |-- part3_sam3_propainter_adapter.py
|   |-- diffueraser_adapter.py
|   |-- check_diffueraser_setup.py
|   `-- part3_ablation.py
|-- assets/                           # Flowcharts, metric summaries, qualitative figures
|-- sample_results/part12_final/       # Curated Part 1 / Part 2 result videos
|-- sample_results/part3_sam3/         # Lightweight Part 3 SAM3 preview videos
|-- RESULTS_MANIFEST.md                # Index of included result videos and metrics
|-- Report.pdf                         # Final compiled report
|-- data/
|   |-- sample/
|   `-- wild/
`-- outputs/                          # Example outputs / generated results when present
```


## Included Results and Report Files

For a quick index of included outputs, see `RESULTS_MANIFEST.md`.

- `sample_results/part12_final/`: representative Part 1 and Part 2 processed videos for bmx-trees, tennis, and wild gymnastics.
- `sample_results/part3_sam3/`: representative Part 3 SAM3 / DiffuEraser preview videos.
- `assets/`: metric CSV/JSON files, DAVIS summaries, qualitative comparison figures, and flowcharts.
- `Report.pdf`: final compiled report.
- `report_final.tex`: final CVPR-template report source.
- `main.bib`, `cvpr.sty`, `preamble.tex`, `ieeenat_fullname.bst`: LaTeX support files for compiling the report.

The full generated frame folders are intentionally excluded from GitHub. They can be regenerated from the scripts and configs.

## Method Flowcharts

### Overall Roadmap

![Overall roadmap](assets/flowchart_overview.png)

### Part 1: Hand-Crafted Baseline

![Part 1 flowchart](assets/flowchart_part1.png)

### Part 2: SAM2 + ProPainter

![Part 2 flowchart](assets/flowchart_part2.png)

### Part 3: Mask Refinement and Ablation

![Part 3 flowchart](assets/flowchart_part3.png)

## Method Summary

### Part 1: Traditional Baseline

`part1_pipeline.py` implements a transparent baseline:

- Detect candidate object regions with YOLOv8 segmentation when available.
- Estimate motion with sparse and dense optical flow.
- Keep regions that are consistent with dynamic foreground motion.
- Apply temporal persistence and morphology to stabilize masks.
- Use two masks: a tighter evaluation mask in `masks/`, and a larger inpainting mask in `inpaint_masks/` to reduce visible object residues.
- Remove the target region using OpenCV inpainting.

### Part 2: SOTA Reproduction

`part2_pipeline.py` runs the main SOTA-style pipeline:

- YOLO detects prompt boxes on selected frames.
- SAM2 propagates masks through the video.
- Tight SAM2 masks are saved to `masks/` for JM/JR evaluation.
- Enlarged masks are saved to `inpaint_masks/` and sent to ProPainter for cleaner object removal.
- ProPainter produces the final restored video.

The final Part 2 setting targets dynamic activity objects such as people, bicycles, sports balls, and tennis rackets, because the project asks for removing the moving object/activity rather than only the visible human body.

### Part 3: Adaptive Mask Refinement

`part3_pipeline.py` is a standalone Part 3 runner. It still uses SAM2 and ProPainter as external backends, but inserts an explicit refinement stage between raw mask generation and inpainting:

- Generate broad raw SAM2 masks for dynamic objects.
- Estimate optical-flow motion maps.
- Use motion-core filtering to suppress static background false positives.
- Adapt the motion threshold according to raw-mask area.
- Maintain target identity with centroid, velocity, and overlap continuity.
- Apply temporal recovery to avoid sudden under-masking.
- Build a slightly larger inpainting mask for ProPainter.

The recommended Part 3 config is `part3_adaptive_motion_refinement.yaml`. Older Part 3 configs are kept as baseline variants.

## Installation

Create and activate a Python environment first. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA, install the PyTorch build that matches your GPU/CUDA version before running SAM2 + ProPainter.

## External Dependencies for Part 2 / Part 3

Part 1 can run with the base dependencies. Part 2 and Part 3 require SAM2 and ProPainter.

### Install SAM2

```powershell
mkdir third_party
cd third_party
git clone https://github.com/facebookresearch/sam2.git
cd sam2
$env:SAM2_BUILD_CUDA='0'
python -m pip install -e .
cd ..\..
```

If CUDA extension compilation works on your machine, `SAM2_BUILD_CUDA=0` can be omitted.

### Install ProPainter

```powershell
cd third_party
git clone https://github.com/sczhou/ProPainter.git
cd ProPainter
python -m pip install -r requirements.txt
cd ..\..
```

Download ProPainter pretrained weights according to the official ProPainter instructions. If ProPainter is stored outside this repository, set:

```powershell
$env:PROPAINTER_REPO='D:\path\to\ProPainter'
```

For 8GB GPUs, use low-memory ProPainter settings:

```powershell
$env:PROPAINTER_EXTRA_ARGS='--resize_ratio 0.5 --subvideo_length 30 --neighbor_length 6 --raft_iter 12 --ref_stride 15'
```

## Model Weights

Model weights are not committed:

- YOLOv8 weights are downloaded automatically by `ultralytics` on first use.
- SAM2 checkpoints are downloaded through the Hugging Face cache when `SAM2VideoPredictor.from_pretrained(...)` is called.
- ProPainter weights should be downloaded following the official ProPainter repository.

## How to Run

All commands should be run from the repository root.

### Unified Part 2 / Part 3 Runner

The recommended current entry point for Part 2 and Part 3 is `part23_integrated_pipeline.py`.
It keeps the stable Part 2 baseline and the Part 3 adaptive refinement in one interface.

Part 2 baseline:

```powershell
python part23_integrated_pipeline.py `
  --mode part2 `
  --input "data/sample/bmx-trees.mp4" `
  --output-dir "outputs\part2_bmx_unified" `
  --device cuda:0 `
  --fallback-opencv
```

Part 2 motion-strict variant:

```powershell
python part23_integrated_pipeline.py `
  --mode part2_motion_strict `
  --input "data/sample/bmx-trees.mp4" `
  --output-dir "outputs\part2_bmx_motion_strict" `
  --device cuda:0 `
  --fallback-opencv
```

Part 3 adaptive refinement:

```powershell
python part23_integrated_pipeline.py `
  --mode part3_adaptive `
  --input "data/sample/bmx-trees.mp4" `
  --output-dir "outputs\part3_bmx_adaptive" `
  --device cuda:0 `
  --fallback-opencv
```

The script automatically selects a mode-specific config when `--config` is omitted:
`part2_example.yaml`, `part2_motion_strict.yaml`, or `part3_adaptive_motion_refinement.yaml`.

### Part 1 Example

```powershell
python part1_pipeline.py `
  --input "data\sample\bmx-trees.mp4" `
  --output-dir "outputs\part1_bmx_trees" `
  --device cuda:0 `
  --target-classes "person,bicycle,sports ball,tennis racket" `
  --motion-thresh 1.0 `
  --dense-motion-thresh 0.8 `
  --dilate 1 `
  --close-ksize 3 `
  --open-ksize 3 `
  --temporal-radius 30 `
  --inpaint-dilate-extra 11
```

Main outputs:

- `outputs/.../dynamic_mask.mp4`
- `outputs/.../inpaint_mask.mp4`
- `outputs/.../inpainted.mp4`
- `outputs/.../masks/`
- `outputs/.../inpaint_masks/`
- `outputs/.../run_meta.json`

### Part 2 Example: SAM2 + ProPainter

```powershell
python part2_pipeline.py `
  --input "data\sample\bmx-trees.mp4" `
  --output-dir "outputs\part2_bmx_trees" `
  --mask-backend sam2 `
  --inpaint-backend propainter `
  --config "configs\part2_example.yaml" `
  --device cuda:0 `
  --inpaint-mask-dilate-extra 11 `
  --inpaint-mask-close 5
```

Main outputs:

- `outputs/.../part2_masks.mp4`
- `outputs/.../part2_inpaint_masks.mp4`
- `outputs/.../part2_inpainted.mp4`
- `outputs/.../masks/`
- `outputs/.../inpaint_masks/`
- `outputs/.../inpaint_frames/`
- `outputs/.../part2_run_meta.json`

### Part 3 Example

```powershell
python part3_pipeline.py `
  --input "data\sample\davis_extra\davis_breakdance.mp4" `
  --output-dir "outputs\part3_breakdance_adaptive" `
  --mask-backend sam2 `
  --inpaint-backend propainter `
  --config "configs\part3_adaptive_motion_refinement.yaml" `
  --device cuda:0 `
  --motion-core-percentile 85 `
  --motion-core-dilate 25 `
  --area-drop-ratio 0.45 `
  --temporal-recovery-dilate 81 `
  --inpaint-mask-dilate-extra 13 `
  --inpaint-mask-close 5
```

Part 3 main outputs:

- `outputs/.../part3_raw_masks.mp4`
- `outputs/.../part3_masks.mp4`
- `outputs/.../part3_inpaint_masks.mp4`
- `outputs/.../part3_inpainted.mp4`
- `outputs/.../part3_run_meta.json`

### Part 3 Ablation Example

After a Part 3 run has produced `raw_masks/`, the ablation script compares raw SAM2, motion-core only, motion-core + temporal recovery, and full adaptive identity refinement without rerunning SAM2 or ProPainter:

```powershell
python tools\part3_ablation.py `
  --input-video "data\sample\davis_extra\davis_breakdance.mp4" `
  --raw-mask-dir "outputs\part3_breakdance_adaptive\raw_masks" `
  --output-dir "outputs\part3_breakdance_ablation" `
  --output-figure "assets\part3_breakdance_ablation.png" `
  --frames "8,24,42,62"
```


### Part 3 Additional Branch: SAM3 + ProPainter / DiffuEraser

The teammate Part 3 branch is integrated as a separate, clearly named route:

```text
part3_sam3_pipeline.py
configs/part3_sam3_*.yaml
tools/sam3_auto_mask.py
tools/diffueraser_adapter.py
tools/part3_sam3_propainter_adapter.py
```

This branch supports:

- `--mask-backend sam3`
- `--inpaint-backend propainter`
- `--inpaint-backend diffueraser`

Example command for the SAM3 + ProPainter route:

```powershell
python part3_sam3_pipeline.py `
  --input "data/sample/tennis.mp4" `
  --output-dir "outputs\part3_sam3_tennis_propainter" `
  --mask-backend sam3 `
  --inpaint-backend propainter `
  --config "configs\part3_sam3_propainter_part2style.yaml" `
  --device cuda:0 `
  --fallback-opencv
```

Example command for the SAM3 + DiffuEraser route:

```powershell
python part3_sam3_pipeline.py `
  --input "data/sample/tennis.mp4" `
  --output-dir "outputs\part3_sam3_tennis_diffueraser" `
  --mask-backend sam3 `
  --inpaint-backend diffueraser `
  --config "configs\part3_sam3_diffueraser_part2style.yaml" `
  --device cuda:0 `
  --fallback-opencv
```

The final report should use the existing result summaries in `assets/`; rerunning
SAM3 or DiffuEraser is optional and requires separate local installations. See
`PART3_INTEGRATION.md` for the relationship between the adaptive-refinement
Part 3 branch and the SAM3/DiffuEraser branch.

Lightweight qualitative videos from the SAM3 branch are stored in:

```text
sample_results/part3_sam3/
```

## Dataset Mapping

Mandatory datasets used in this project:

- Wild Video: `data/wild/gymnastics_easy_long_12s_720p.mp4`
- Sample Data: `data/sample/bmx-trees.mp4`
- Sample Data: `data/sample/tennis.mp4`

Additional optional DAVIS 2017 examples for qualitative stress testing:

- Easy human motion: `data/sample/davis_extra/davis_walking.mp4`
- Human-object interaction: `data/sample/davis_extra/davis_bike-packing.mp4`
- Fast non-rigid motion: `data/sample/davis_extra/davis_breakdance.mp4`

DAVIS masks can be used when available for quantitative mask evaluation. The optional MP4 examples above are included as lightweight qualitative examples; quantitative JM/JR should only be reported when matching GT masks are available.

## Visual Results

The following figures compare original frames, ground-truth masks when available, Part 1 masks/results, and Part 2 masks/results.

### BMX-Trees

![BMX qualitative results](assets/bmx_results.png)

### Tennis

![Tennis qualitative results](assets/tennis_results.png)

### Wild Gymnastics

![Wild gymnastics qualitative results](assets/wild_gymnastics_results.png)

### Part 3 Mask Refinement Ablation

![Part 3 BMX ablation](assets/part3_bmx_ablation.png)

### Part 3 DAVIS Breakdance Refinement

![Part 3 breakdance final comparison](assets/part3_breakdance_final_comparison.png)

### Part 3 DAVIS Breakdance Ablation

![Part 3 breakdance ablation](assets/part3_breakdance_ablation.png)

The ablation CSV is stored at `assets/part3_breakdance_ablation_summary.csv`.

## Quantitative Results

The course update states that quantitative metrics are not required for mandatory datasets that do not provide ground truth. Therefore:

- For datasets with available GT masks, we report JM/JR.
- For datasets without GT masks, we report qualitative comparisons only.
- PSNR/SSIM should only be used when aligned ground-truth restored frames are available.

Latest Part 1 / Part 2 mask results on datasets with GT masks:

| Dataset | Method | Old JM | New JM | Old JR | New JR |
|---|---:|---:|---:|---:|---:|
| bmx-trees | Part 1 | 0.3557 | 0.4098 | 0.2750 | 0.4000 |
| bmx-trees | Part 2 | 0.5056 | 0.6526 | 0.7000 | 0.9250 |
| tennis | Part 1 | 0.6455 | 0.7822 | 0.9857 | 1.0000 |
| tennis | Part 2 | 0.7127 | 0.9342 | 1.0000 | 1.0000 |

The CSV version is stored at `assets/part12_results_summary.csv`.

### Evaluate JM / JR

```powershell
python evaluate_metrics.py `
  --pred-mask-dir "outputs\run_name\masks" `
  --gt-mask-dir "path\to\gt_masks" `
  --recall-thr 0.5
```

### Evaluate PSNR / SSIM

```powershell
python evaluate_metrics.py `
  --pred-frame-dir "outputs\run_name\inpaint_frames" `
  --gt-frame-dir "path\to\gt_frames"
```

## Notes for GitHub Upload

The following are intentionally ignored by git:

- `outputs/`: generated videos, frames, masks, and metadata.
- `third_party/`: local SAM2 and ProPainter repositories.
- `models/` and model weights such as `.pt`, `.pth`, `.ckpt`.
- Python caches and local virtual environments.

Processed videos should be submitted separately through the required course submission system rather than committed with all generated frames.


