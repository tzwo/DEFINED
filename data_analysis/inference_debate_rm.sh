#!/bin/bash

# Simple wrapper to score the debate RM dataset using a trained adapter

export MODEL_PATH="../model/Qwen2.5-7B-Instruct"
export ADAPTER_PATH=""
export TEMPLATE_PATH="qwen2.5-7b.jinja"
export DATA_PATH=""
export OUTPUT_PATH=""

CUDA_VISIBLE_DEVICES=0,1,2,3 python ./inference_debate_rm.py \
  --model_path "$MODEL_PATH" \
  --adapter_path "$ADAPTER_PATH" \
  --template_path "$TEMPLATE_PATH" \
  --example_path "$DATA_PATH" \
  --output_path "$OUTPUT_PATH" \
  --device_ids "0,1,2,3" \
  --batch_size 32