#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-$PROJECT_ROOT}"
SOURCE_ZIP="${SOURCE_ZIP:-/tmp/AIAA3201_project_main_wild.zip}"

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "[bootstrap] wild source not found at $SOURCE_ROOT"
  echo "[bootstrap] expected bundled source at $PROJECT_ROOT/baselines/main_branch"
  PROJECT_ROOT="$PROJECT_ROOT" SOURCE_ROOT="$SOURCE_ROOT" SOURCE_ZIP="$SOURCE_ZIP" python3 - <<'PY'
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
source_root = Path(os.environ["SOURCE_ROOT"]).resolve()
source_zip = Path(os.environ["SOURCE_ZIP"]).resolve()
repo_url = "https://github.com/Tacit-Wuyan/AIAA3201_project/archive/refs/heads/main.zip"

source_root.parent.mkdir(parents=True, exist_ok=True)
print(f"[bootstrap] downloading {repo_url} -> {source_zip}")
urllib.request.urlretrieve(repo_url, source_zip)

tmpdir = Path(tempfile.mkdtemp(prefix="aiaa3201_wild_"))
with zipfile.ZipFile(source_zip, "r") as zf:
    zf.extractall(tmpdir)

extracted_root = tmpdir / "AIAA3201_project-main"
if source_root.exists():
    shutil.rmtree(source_root)
shutil.copytree(extracted_root, source_root)
shutil.rmtree(tmpdir)
print(f"[bootstrap] source ready at {source_root}")
PY
fi

PART1_PYTHON="${PART1_PYTHON:-python3}"
PART2_PYTHON="${PART2_PYTHON:-python3}"
PART3_SAM3_PROPAINTER_PYTHON="${PART3_SAM3_PROPAINTER_PYTHON:-python3}"
PART3_SAM2_MASK_PYTHON="${PART3_SAM2_MASK_PYTHON:-python3}"
PART3_DIFFUERASER_PYTHON="${PART3_DIFFUERASER_PYTHON:-python3}"

exec python3 "$PROJECT_ROOT/run_wild_part123_batch.py" \
  --source-root "$SOURCE_ROOT" \
  --part1-python "$PART1_PYTHON" \
  --part2-python "$PART2_PYTHON" \
  --part3-sam3-propainter-python "$PART3_SAM3_PROPAINTER_PYTHON" \
  --part3-sam2-mask-python "$PART3_SAM2_MASK_PYTHON" \
  --part3-diffueraser-python "$PART3_DIFFUERASER_PYTHON" \
  "$@"
