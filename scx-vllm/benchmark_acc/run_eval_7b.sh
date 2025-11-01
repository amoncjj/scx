#!/usr/bin/env bash
# ==============================================
#  LM Evaluation harness + vLLM + SCX 环境封装脚本
#  用法: bash run_eval.sh [TASK] [OUTPUT_DIR]
#  例如: bash run_eval.sh lambada_openai results_lambada
# ==============================================

export HF_ENDPOINT=https://hf-mirror.com

# 默认参数
TASK=${1:-lambada_openai}
OUTDIR=${2:-results}

# 模型参数
MODEL_PATH="/data/model_hub/DeepSeek-R1-Distill-Qwen-7B"
DTYPE="bfloat16"

# 输出文件夹
mkdir -p "$OUTDIR"

# 启动评测
echo "[INFO] Running LM Eval on task: $TASK"
echo "[INFO] Results will be saved to: $OUTDIR"

# SCX 环境变量
export SCX_ENC_LAYERS=""
export SCX_SECURE_DEVICE="cpu"
export SCX_ENABLE_DEBUG="True"

# 运行不使用 scx 编码 layers 的测评
lm_eval \
  --model vllm \
  --model_args pretrained="${MODEL_PATH}",tensor_parallel_size=1,enforce_eager=True,dtype=${DTYPE} \
  --tasks "${TASK}" \
  --batch_size auto \
  --output_path "${OUTDIR}"


# SCX 环境变量
export SCX_ENC_LAYERS="0,27"

# 运行带有 SCX 编码 layers 的测评
lm_eval \
  --model vllm \
  --model_args pretrained="${MODEL_PATH}",tensor_parallel_size=1,enforce_eager=True,dtype=${DTYPE} \
  --tasks "${TASK}" \
  --batch_size auto \
  --output_path "${OUTDIR}"

echo "[INFO] Evaluation finished. Check $OUTDIR for results."
