#!/usr/bin/env bash
set -e

TRAIN_FILE=${1:-/path/to/train-v1.1.json}
MODEL_PATH=${2:-./qa_model}

python -m src.eval \
  --train_file "$TRAIN_FILE" \
  --model_path "$MODEL_PATH" \
  --train_size 2000 \
  --valid_size 500



