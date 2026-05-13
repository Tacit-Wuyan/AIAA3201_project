"""Part 3 entry point: mask-refinement exploration pipeline.

Part 3 uses the same execution engine as Part 2, but defaults the output
prefix to ``part3`` so generated files are named part3_inpainted.mp4,
part3_masks.mp4, and part3_run_meta.json.
"""

import sys

from part2_pipeline import main


if __name__ == "__main__":
    if "--output-prefix" not in sys.argv:
        sys.argv.extend(["--output-prefix", "part3"])
    main()
