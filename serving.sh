
#!/bin/bash

MODEL_PATH=saved/fc_cp_7500_fc_json

MODEL_NAME=Qwen3-1.7B-FC
PORT=8268

CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server --model $MODEL_PATH --served-model-name $MODEL_NAME --max-model-len 16384 --gpu-memory-utilization 0.9 --trust-remote-code --enable-auto-tool-choice --tool-call-parser hermes --port $PORT
