#!/bin/bash
# Debug LoRA checkpoint compatibility with SGLang

set -e

CHECKPOINT_PATH=${1:-""}

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: ./scripts/debug_lora_checkpoint.sh <checkpoint_path>"
    echo "Example: ./scripts/debug_lora_checkpoint.sh ./checkpoints/.../weight_update"
    exit 1
fi

python -m areal.utils.temp_debug_lora "$CHECKPOINT_PATH"