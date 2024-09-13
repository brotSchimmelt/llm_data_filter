#!/bin/bash
set -e

# set and create the outlines cache directory
UNIQUE_ID=$(date +%Y%m%d%H%M%S)_$RANDOM
CACHE_DIR="$HOME/.cache/outlines_custom_cache/job_$UNIQUE_ID"
export OUTLINES_CACHE_DIR="$CACHE_DIR"
echo "Using cache directory: $CACHE_DIR"
mkdir -p "$CACHE_DIR"

start_time=$(date +%s)

export LOG_LEVEL="WARNING" # used for vLLM logging
export IC_DEBUG="False"  # icecream print debugging
export TOKENIZERS_PARALLELISM="false"  # for sentence-transformers


python filter_dataset.py --model_name mistral

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
hours=$((elapsed_time / 3600))
minutes=$(( (elapsed_time % 3600) / 60 ))
seconds=$((elapsed_time % 60))
printf "All experiments completed. Total runtime: %02d:%02d:%02d\n" $hours $minutes $seconds

# clean up the custom outlines cache directory
if [ -d "$CACHE_DIR" ]; then
    echo "Cleaning up cache directory: $CACHE_DIR"
    rm -rf "$CACHE_DIR"
else
    echo "Cache directory not found: $CACHE_DIR"
fi
