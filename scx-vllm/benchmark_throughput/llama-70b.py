import argparse

parser = argparse.ArgumentParser(description="SCX ENV layer configuration")
parser.add_argument('--scx_enc_layers', type=str, default="",
                    help="Specify SCX_ENC_LAYERS (e.g., '0', '0,1,2', etc.)")
args = parser.parse_args()

# ===== before calling vllm, call scx_env_init to initialize the scx environment =====
from scx.keys import scx_env_init

# 28 layers qwen2-1.5b
SCX_ENC_LAYERS = args.scx_enc_layers
SCX_SECURE_DEVICE = "cpu"
SCX_ENABLE_DEBUG = True
scx_env_init(SCX_ENC_LAYERS, SCX_SECURE_DEVICE, SCX_ENABLE_DEBUG)
# ===== then you can call vllm just like normal ======

from vllm import LLM, SamplingParams

model_path = "/data/model_hub/DeepSeek-R1-Distill-Llama-70B"
MAX_MODEL_LEN = 38000 # 38k，32k输入，4k输出，留下一些冗余
DTYPE = "bfloat16"

llm = LLM(
    model=model_path,   
    dtype=DTYPE,                   
    tensor_parallel_size=8,        # 8卡并行
    max_model_len=MAX_MODEL_LEN,   
    gpu_memory_utilization=0.92,   # 充分利用 48GB 显存（按需微调 0.85~0.95）
    enforce_eager=True,           
)

for ISL in ["1k", "4k", "8k", "32k"]:
    for OSL in [512, 1024, 2048, 4096]:
        print(f"----- ISL: {ISL}, OSL: {OSL}  Start -----")
        prompt = open(f"/home/yuanmu/scx-huawei-exp/data/prompt_{ISL}.txt", "r", encoding="utf-8").read()
        sampling_params = SamplingParams(
            min_tokens=OSL,
            max_tokens=OSL, 
            temperature=0.0,
            top_k=1,
            top_p=1.0,
        )
        outputs = llm.generate([prompt], sampling_params)