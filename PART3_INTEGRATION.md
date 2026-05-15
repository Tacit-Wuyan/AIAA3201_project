# Part 3 Integration Notes

This repository contains two complementary Part 3 branches.

## `part3_pipeline.py`: Adaptive Motion Refinement

Our original Part 3 branch refines SAM2 masks with motion-core filtering,
temporal identity support, and adaptive mask post-processing. It is useful for
failure cases such as `breakdance`, where raw masks can over-segment static
background.

## `part3_sam3_pipeline.py`: SAM3 / DiffuEraser Branch

The teammate branch adds a separate Part 3 implementation with:

- `tools/sam3_auto_mask.py` for SAM3-based open-vocabulary mask generation.
- `tools/diffueraser_adapter.py` for DiffuEraser inpainting.
- `tools/part3_sam3_propainter_adapter.py` for the ProPainter route used by this branch.
- `configs/part3_sam3_*.yaml` for SAM3 + ProPainter / SAM3 + DiffuEraser settings.
- `run_davis_sam_ablation_batch.py` and `run_wild_part123_batch.py` for batch evaluation.

The existing result summaries in `assets/` should be used for the final report;
no rerun is required for submission. To rerun this branch, install SAM3 and/or
DiffuEraser separately and point the scripts to the correct environment.

## Result Interpretation

The current DAVIS ablation shows that SAM3 improves robustness on some failed
SAM2 cases, but it does not universally improve mean mask accuracy over the
SAM2-based Part 2 pipeline. This is why the report should present SAM3 as a
Part 3 exploration rather than a replacement for Part 2.
