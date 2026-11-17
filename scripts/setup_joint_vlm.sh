#!/usr/bin/env bash
set -euo pipefail
curio_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$curio_root"
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This local worker requires Apple Silicon. The scoring interface can be implemented by another backend on CUDA hosts."
  exit 1
fi
export UV_CACHE_DIR="$curio_root/.cache/uv-vlm"
if [[ ! -x .venv-vlm/bin/python ]]; then
  uv venv --python .venv/bin/python .venv-vlm
fi
uv pip install --python .venv-vlm/bin/python -r requirements-vlm.txt
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=180 .venv-vlm/bin/python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
name = 'mlx-community/Qwen3-VL-8B-Instruct-4bit'
revision = 'defcdea7cc7a4b0858fea563cbbce171d328e457'
path = snapshot_download(name, revision=revision, local_dir='models/vlm_qwen3_8b_4bit',
                         allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model','README.md'])
out = Path('artifacts/joint_vlm_v9'); out.mkdir(parents=True, exist_ok=True)
(out/'model_receipt.json').write_text(json.dumps({'repository':name,'revision':revision,'local_dir':str(path)},indent=2)+'\n')
PY
uv pip freeze --python .venv-vlm/bin/python > artifacts/joint_vlm_v9/environment.lock.txt
