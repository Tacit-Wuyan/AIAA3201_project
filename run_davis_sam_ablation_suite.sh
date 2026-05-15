#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-$PROJECT_ROOT}"
SOURCE_ZIP="${SOURCE_ZIP:-/tmp/AIAA3201_project_main.zip}"

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "[bootstrap] main branch source not found at $SOURCE_ROOT"
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

tmpdir = Path(tempfile.mkdtemp(prefix="aiaa3201_main_"))
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

SAM3_PROPAINTER_PYTHON="${SAM3_PROPAINTER_PYTHON:-python3}"
SAM2_DIFFUERASER_PYTHON="${SAM2_DIFFUERASER_PYTHON:-python3}"
SAM2_DIFFUERASER_INPAINT_PYTHON="${SAM2_DIFFUERASER_INPAINT_PYTHON:-python3}"
SAM3_DIFFUERASER_PYTHON="${SAM3_DIFFUERASER_PYTHON:-python3}"

exec python3 "$PROJECT_ROOT/run_davis_sam_ablation_batch.py" \
  --source-root "$SOURCE_ROOT" \
  --sam3-propainter-python "$SAM3_PROPAINTER_PYTHON" \
  --sam2-diffueraser-python "$SAM2_DIFFUERASER_PYTHON" \
  --sam2-diffueraser-inpaint-python "$SAM2_DIFFUERASER_INPAINT_PYTHON" \
  --sam3-diffueraser-python "$SAM3_DIFFUERASER_PYTHON" \
  "$@"
