#!/usr/bin/env bash
# run.sh - run the full pipeline scripts in order
# This script executes the main pipeline stages in sequence.

set -euo pipefail

echo "[run.sh] Starting full pipeline run at $(date --iso-8601=seconds)"

python src/00_aggregate_jsons.py
python src/01_preprocess.py
python src/02_train.py
python src/03_evaluation.py
python src/04_inference.py

echo "[run.sh] Pipeline finished at $(date --iso-8601=seconds)"