from transformers import LlamaForCausalLM, LlamaConfig
import torch

from scx.keys import SCXKeyGenerator
from scx.models.llama import encode_llama
from scx.kvcache import split_kvcache_dynamic

device = "cuda:0"
num_hidden_layers = 3
seq_len = 10
hidden_dim = 4096
qk_hidden_dim = 128      # 注意力 head 的总 hidden 维度
redundant_num = 0        # 不使用冗余嵌入
alp = False              # 不加噪（测试用）
batch_size = 1
decode_steps = 5

# 1. 构建 Llama 模型
config = LlamaConfig(
    vocab_size=1000,
    num_hidden_layers=num_hidden_layers,
    hidden_size=hidden_dim,
)
model = LlamaForCausalLM(config).eval().half().to(device)

# 2. 构建 SCX 密钥生成器
key_gen = SCXKeyGenerator(
    seq_len=seq_len,
    hidden_dim=hidden_dim,
    qk_hidden_dim=qk_hidden_dim,
    redundant_num=redundant_num,
    alp=alp,
    batch_size=batch_size,
    decode_start_layers=[0],  # 第 0 层是 decode 的「开始层」
    decode_end_layers=[2],    # 第 2 层是 decode 的「结束层」
)

# 3. 将 SCX 注入 Llama 模型
encode_llama(model, key_gen)

# 4. Prefill 阶段
input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
with torch.no_grad():
    output = model(input_ids, mode="prefill")
    logits = output.logits
    kvcache = output.past_key_values