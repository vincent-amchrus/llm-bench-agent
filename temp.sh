cat serving4.sh 
export HF_HOME=/mnt/data/function_calling/models/opensource
export HF_HUB_CACHE=/mnt/data/function_calling/models/opensource

CUDA_VISIBLE_DEVICES=0 vllm serve unsloth/Qwen3-4B-Instruct-2507  \
  --enable-lora \
  --lora-modules \
  _3012_100k_ckp500=models/_3012_14k_small_talk_16k_ultrachat_synthetic_31k_smart__39k_global_tools_unsloth-Qwen3-4B-Instruct-2507/checkpoint-500 \
  _3012_100k_ckp3000=models/_3012_14k_small_talk_16k_ultrachat_synthetic_31k_smart__39k_global_tools_unsloth-Qwen3-4B-Instruct-2507/checkpoint-3000 \
  _3012_100k_ckp8000=models/_3012_14k_small_talk_16k_ultrachat_synthetic_31k_smart__39k_global_tools_unsloth-Qwen3-4B-Instruct-2507/checkpoint-8000 \
  _3012_100k_ckp12584=models/_3012_14k_small_talk_16k_ultrachat_synthetic_31k_smart__39k_global_tools_unsloth-Qwen3-4B-Instruct-2507/checkpoint-12584 \
  --dtype bfloat16 --host 0.0.0.0 --port 8268 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 16384 \
  --gpu_memory_utilization 0.9 \
  --max-lora-rank 16


CUDA_VISIBLE_DEVICES=0 vllm serve unsloth/Qwen3-4B-Instruct-2507  \
  --enable-lora \
  --lora-modules \
  _3012_100k_ckp500=models/_3012_14k_small_talk_16k_ultrachat_synthetic_31k_smart__39k_global_tools_unsloth-Qwen3-4B-Instruct-2507/checkpoint-500 \
  _3012_100k_ckp3000=models/_3012_14k_small_talk_16k_ultrachat_synthetic_31k_smart__39k_global_tools_unsloth-Qwen3-4B-Instruct-2507/checkpoint-3000 \
  _3012_100k_ckp8000=models/_3012_14k_small_talk_16k_ultrachat_synthetic_31k_smart__39k_global_tools_unsloth-Qwen3-4B-Instruct-2507/checkpoint-8000 \
  _3012_100k_ckp12584=models/_3012_14k_small_talk_16k_ultrachat_synthetic_31k_smart__39k_global_tools_unsloth-Qwen3-4B-Instruct-2507/checkpoint-12584 \
  --dtype bfloat16 --host 0.0.0.0 --port 8268 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --max-model-len 16384 \
  --gpu_memory_utilization 0.9 \
  --max-lora-rank 16