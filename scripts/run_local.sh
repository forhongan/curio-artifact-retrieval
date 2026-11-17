#!/usr/bin/env bash
set -euo pipefail
curio_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$curio_root"
export CURIO_ENCODER="${CURIO_ENCODER:-siglip}"
export CURIO_INDEX="${CURIO_INDEX:-artifacts/siglip_adapted}"
export CURIO_DATA="${CURIO_DATA:-data/snapshots/collection_v2.jsonl}"
export CURIO_PROBE_WEIGHT="${CURIO_PROBE_WEIGHT:-0}"
export CURIO_HOST="${CURIO_HOST:-127.0.0.1}"
export CURIO_PORT="${CURIO_PORT:-8000}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$curio_root/models/huggingface/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="$curio_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$curio_root/.venv/bin/python" -m curio.serve
