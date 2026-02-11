#!/bin/bash

MODEL_PATH=/mnt/data/models/Qwen3-Next-80B-A3B-Instruct
MODEL_NAME=Qwen/Qwen3-Next-80B-A3B-Instruct
PORT=8006
API_KEY=token-vllm
TENSOR_PARALLEL_SIZE=4 # increase when model is too large, cannot fit weight into one GPU
DATA_PARALLEL_SIZE=1 # increase when we want to increase the throughput
MAX_MODEL_LEN=131072
GPU_MEM_UTIL=0.8

python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --served-model-name $MODEL_NAME --trust-remote-code --data_parallel_size $DATA_PARALLEL_SIZE --tensor_parallel_size $TENSOR_PARALLEL_SIZE --port $PORT --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEM_UTIL --enable-auto-tool-choice --tool-call-parser hermes --api-key $API_KEY