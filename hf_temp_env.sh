#!/bin/bash

export HF_HOME=/tmp/emma_hf
export HF_DATASETS_CACHE=/tmp/emma_hf/datasets
export TRANSFORMERS_CACHE=/tmp/emma_hf/transformers

mkdir -p "$HF_HOME"
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$TRANSFORMERS_CACHE"

echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
