# New Results 2026-05-15 Summary

## DAVIS SAM Ablation
- sam2_diffueraser: success=48/50, failed=2, mean JM=0.742897, mean JR=0.823891
- sam3_diffueraser: success=50/50, failed=0, mean JM=0.683948, mean JR=0.74099
- sam3_propainter: success=50/50, failed=0, mean JM=0.683948, mean JR=0.74099
- SAM3 vs SAM2 on 48 common successful sequences: mean delta JM=-0.056425, mean delta JR=-0.093693; SAM3 better on 17 sequences and worse on 30 sequences.

## Wild Video
- part1_main: ok, runtime=759.725s
- part2_main_propainter: failed, runtime=41.437s
- part3_sam2_diffueraser: ok, runtime=447.037s
- part3_sam3_propainter: ok, runtime=433.838s
- part3_sam3_diffueraser: ok, runtime=560.829s
- Note: no GT for wild video, so use qualitative visual comparison only.
- Warning: `part2_main_propainter` failed in this package due to a path separator issue; do not use this zip as the sole final Part 2 wild evidence.