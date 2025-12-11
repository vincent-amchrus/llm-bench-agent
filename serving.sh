CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-1.7B  \
  --enable-lora \
  --lora-modules \
  checkpoint_180=models/checkpoint-180 \
  checkpoint_45=models/checkpoint-45 \
  --dtype bfloat16 --host 0.0.0.0 --port 8268 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 16384 \
  --gpu_memory_utilization 0.7 \
  --max-lora-rank 32