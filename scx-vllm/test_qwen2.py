# ===== before calling vllm, call scx_env_init to initialize the scx environment =====
from scx.keys import scx_env_init

# 28 layers qwen2-1.5b
SCX_ENC_LAYERS = "0"
SCX_SECURE_DEVICE = "cpu"
SCX_ENABLE_DEBUG = True
scx_env_init(SCX_ENC_LAYERS, SCX_SECURE_DEVICE, SCX_ENABLE_DEBUG)
# ===== then you can call vllm just like normal ======

from vllm import LLM, SamplingParams

model_path = "/home/yuanmu/models/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_MODEL_LEN = 38000 # 38k，32k输入，4k输出，留下一些冗余
DTYPE = "bfloat16"

llm = LLM(
    model=model_path,   
    dtype=DTYPE,                   
    tensor_parallel_size=1,        # 单卡即可
    max_model_len=MAX_MODEL_LEN,   
    gpu_memory_utilization=0.92,   # 充分利用 48GB 显存（按需微调 0.85~0.95）
    enforce_eager=True,           # NOTE 开发时设置为 True，不然无法打印调试
)

OSL = 512
prompt = "Hello, how are you?"
sampling_params = SamplingParams(
    min_tokens=OSL,
    max_tokens=OSL, 
    temperature=0.0,
    top_k=1,
    top_p=1.0,
    stop=[], # 禁止任何停止符
    ignore_eos=True,
)
outputs = llm.generate([prompt], sampling_params)
for output in outputs:
    print("Generated text:", output.outputs[0].text)

"""
NO SCX:
Generated text:  I have a question about the math. I need to find the value of the integral ∫₀^π sin²x dx. I need to use the substitution method. Please help me. I'm a bit confused about how to approach this.

Alright, let me try to figure this out. I remember that when integrating trigonometric functions, especially squared ones, there are some identities that can simplify the problem. Maybe I can use the power-reduction formula or something like that. But I'm not entirely sure how to apply it here.

Wait, the integral is from 0 to π, and the integrand is sin²x. I think the power-reduction formula for sin²x is something like sin²x = (1 - cos2x)/2. Yeah, that sounds right. So if I substitute that into the integral, I can rewrite it as ∫₀^π (1 - cos2x)/2 dx. That should make the integral easier to handle.

Okay, so let me write that down: ∫₀^π (1 - cos2x)/2 dx. I can factor out the 1/2 to make it simpler: (1/2) ∫₀^π (1 - cos2x) dx. Now, I can split this integral into two separate integrals: (1/2) [ ∫₀^π 1 dx - ∫₀^π cos2x dx ]. That should make it easier to compute each part individually.

Let's tackle the first integral: ∫₀^π 1 dx. That's straightforward. The integral of 1 with respect to x is just x, so evaluating from 0 to π gives π - 0 = π. So the first part is π.

Now, the second integral: ∫₀^π cos2x dx. I remember that the integral of cos(ax) is (1/a) sin(ax) + C. So applying that here, the integral of cos2x is (1/2) sin2x. Therefore, evaluating from 0 to π gives (1/2) sin(2π) - (1/2) sin(0). But sin(2π) is 0 and sin(0) is also 0, so this entire integral is 0 - 0 = 0.

So putting it all together, the original integral becomes (1/2) [ π - 0 ] = (1/2) π. Therefore,

SCX:
Generated text:  I have a question about the math. I need to find the value of the integral ∫₀^π sin²x dx. I need to use the substitution method. Please help me. I'm a bit confused about how to approach this.

Alright, let me try to figure this out. I remember that when integrating trigonometric functions, especially squared ones, there are some identities that can simplify the problem. Maybe I can use the power-reduction formula or something like that. But I'm not entirely sure how to apply it here.

Wait, the integral is from 0 to π, and the integrand is sin²x. I think the power-reduction formula for sin²x is something like sin²x = (1 - cos2x)/2. Yeah, that sounds right. So if I substitute that into the integral, I can rewrite it as ∫₀^π (1 - cos2x)/2 dx. That should make the integral easier to handle.

Okay, so let me write that down: ∫₀^π (1 - cos2x)/2 dx. I can factor out the 1/2 to make it simpler: (1/2) ∫₀^π (1 - cos2x) dx. Now, I can split this integral into two separate integrals: (1/2) [ ∫₀^π 1 dx - ∫₀^π cos2x dx ]. That should make it easier to compute each part individually.

Let's tackle the first integral: ∫₀^π 1 dx. That's straightforward. The integral of 1 with respect to x is just x, so evaluating from 0 to π gives π - 0 = π. So the first part is π.

Now, the second integral: ∫₀^π cos2x dx. I remember that the integral of cos(ax) is (1/a) sin(ax) + C. So applying that here, the integral of cos2x is (1/2) sin2x. Therefore, evaluating from 0 to π gives (1/2) sin(2π) - (1/2) sin(0). But sin(2π) is 0 and sin(0) is also 0, so this entire integral is 0 - 0 = 0.

So putting it all together, the original integral becomes (1/2) [ π - 0 ] = (1/2) π. Therefore,
"""


""""
est. speed input: 165.09 toks/s, output: 82.06 toks/s

speed input: 179.09 toks/s, output: 89.02 toks/s
"""