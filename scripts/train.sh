#!/usr/bin/env bash
set -e

TRAIN_FILE=${1:-/path/to/train-v1.1.json}
OUTPUT_DIR=${2:-./qa_model}

python -m src.train \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --train_size 2000 \
  --valid_size 500 \
  --model_name albert-base-v2



