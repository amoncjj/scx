## SCX：云端保密 Transformer 推理的无状态 KV-Cache 编码（标准实现）

本仓库提出的 **SCX（Stateless KV-Cache Encoding）**，通过对 Transformer 模型中间状态（尤其是 KV Cache）进行随机置换与编码，在保持模型精度和性能基本不变的前提下，降低云服务提供方从中间激活中恢复隐私信息的风险。

本文件主要介绍 **非 vLLM 的标准实现部分**，对应代码目录为 `scx/`，基于 Hugging Face `transformers` 的 Llama 模型实现，可单独使用，不依赖 vLLM。

---

## 目录

- [背景与目标](#背景与目标)
- [核心模块概览（非 vLLM）](#核心模块概览非-vllm)
- [基础使用示例](#基础使用示例)
- [通信 / 执行框架](#通信--执行框架)

---

## 背景与目标

在云端推理场景下，Transformer 模型的中间状态（隐藏层输出、KV Cache 等）可能携带大量敏感信息。传统推理服务通常直接在 GPU 上保存和更新这些状态，一旦云侧发生泄露，中间激活可能被用于还原输入数据或训练数据。

**SCX 的设计目标是：**

- 通过 **随机置换、冗余嵌入、加性噪声** 等方式，对中间张量进行编码；
- 在 **推理 API 和最终输出尽量不变** 的前提下，使得云服务方即使拿到中间 KV Cache，也难以还原原始语义；
- 支持在 **CPU 与 GPU 间分层部署**：部分层在 CPU 侧进行编码/解码，部分层在 GPU 侧高效计算。

---

## 核心模块概览（非 vLLM）

### `scx.keys.SCXKeyGenerator`：SCX 密钥生成器

负责为每一层生成一组随机的「密钥」，包括置换、冗余嵌入和解码阶段的噪声参数：

- **序列维度置换**：
  - `attn_pi_left` / `attn_inv_pi_left`
  - `ff_pi_left` / `ff_inv_pi_left`
  - 用于在 `seq_len` 维度打乱 / 恢复 token 顺序。
- **隐藏维度置换**：
  - `attn_pi_right` / `attn_inv_pi_right`
  - 用于在 attention 的 `hidden_dim` / head 维度中打乱 / 恢复。
- **输出/残差相关置换**：
  - `wo_pi_left` / `wo_inv_pi_left`
  - 在 attention 输出后，再对序列维度进行置换 / 还原。
- **冗余嵌入 `r_embeds`**：
  - 可选地在序列末尾拼接若干个冗余 token，增加混淆。
- **decode 阶段的噪声与置换**：
  - `dec_attn_alp`：加性噪声；
  - `dec_attn_pi_right` / `dec_attn_inv_pi_right`：解码阶段 hidden 维度置换。

主要初始化参数：

- `seq_len`, `hidden_dim`, `qk_hidden_dim`
- `redundant_num`：冗余 token 数；
- `alp`：是否启用加性噪声；
- `batch_size`
- `decode_start_layers` / `decode_end_layers`：指定哪些层是 decode 阶段的开始 / 结束层。

### `scx.models.llama.encode_llama`：将 SCX 注入 Llama 模型

该函数对传入的 `LlamaForCausalLM` 执行「就地改造」：

- 遍历 `model.model.layers` 中的每一层 `LlamaDecoderLayer`：
  - 创建对应的 `SCXLlamaDecoderLayer`；
  - 使用 `copy_submodules` 将原始层的参数完整拷贝到新层；
  - 创建 `SCXLlamaAttention` 替换原先的 `self_attn`，并注入本层的 SCX 密钥；
  - 将模型中的这一层替换为新的 SCX 版本。

效果：在 **不改变外部调用方式** 的前提下，为模型内部注入 SCX 编码逻辑。

### `SCXLlamaDecoderLayer`：带 SCX 的 Decoder 层

定义在 `scx/models/llama.py` 中，继承自 `LlamaDecoderLayer`。主要改动发生在 `forward` 中：

- **prefill 阶段（`mode="prefill"`）**：
  - 在自注意力前后，对 `hidden_states` 进行 `ff_pi_left` / `ff_inv_pi_left` 序列维度置换；
  - 为了在 CPU 上执行部分编码操作，会将张量在 CPU / GPU 间来回移动；
  - 通过这种方式，在不改变网络结构的前提下，实现对中间表示的打乱。

- **decode 阶段（`mode="decode"`）**：
  - 同样复用该层结构，但通过额外的参数控制执行路径：
    - `cache_position`：解码步的位置；
    - `gpu_kvcache` / `cpu_kvcache`：分设备 KV Cache；
  - 根据层编号是否属于 `decode_start_layers` / `decode_end_layers`，决定是否在 CPU 上执行解码相关的噪声 / 置换逻辑。

### `SCXLlamaAttention`：带 SCX 的自注意力模块

同样定义在 `scx/models/llama.py` 中，是 SCX 的核心实现之一。

#### Prefill 模式（`mode="prefill"`）

1. 接收来自上一层的 `hidden_states`（通常在 GPU 上），将其移动到 CPU；
2. 若启用冗余嵌入，则在序列末尾拼接 `r_embeds`；
3. 在 `seq_len` 维度按 `attn_pi_left` 做随机置换，再将张量移回 GPU；
4. 经 Q/K/V 投影后，再在 CPU 上用 `attn_inv_pi_left` 恢复原始序列顺序（并裁掉冗余部分）；
5. 使用 RoPE（`apply_rotary_pos_emb`）对 Q/K 应用位置编码；
6. 通过 `DynamicCache.update` 写入 KV Cache（仍是统一的 cache 对象）；
7. 在 hidden 维度按 `attn_pi_right` 做置换后回到 GPU，调用 `eager_attention_forward` 或其他实现完成注意力计算；
8. 注意力输出再迁移到 CPU，按 `attn_inv_pi_right` 和 `wo_pi_left` / `wo_inv_pi_left` 完成 hidden 与 seq 维度的还原；
9. 最终返回到原设备，继续执行后续 FFN 计算。

#### Decode 模式（`mode="decode"`）

在 decode 阶段，会利用 `decode_start_layers` / `decode_end_layers` 将不同层分为三类：

- **开始层（start layers）**：
  - 在 CPU 上对 `hidden_states` 加上 `dec_attn_alp` 噪声；
  - 在 hidden 维度上使用 `dec_attn_pi_right` / `dec_attn_inv_pi_right` 做打乱 / 还原；
  - 使用 **CPU 端 KV Cache** 进行读写。

- **结尾层（end layers）**：
  - 同样主要与 CPU 端 KV Cache 交互，用于完成解密 / 还原等收尾操作。

- **中间层**：
  - 直接使用 **GPU 端 KV Cache**，不再额外引入 CPU 侧编码，以减小性能开销。

通过以上区分，可以在 **关键层用 CPU 做安全敏感操作**，而其余层仍然利用 GPU 的高算力。

### `scx.kvcache.split_kvcache_dynamic`：KV Cache 跨设备拆分

prefill 阶段结束后，模型返回的 `past_key_values` 是一个 `DynamicCache`，包含所有层的 KV 状态。  
`split_kvcache_dynamic` 将其拆分为两份新的 `DynamicCache`：

- `gpu_kvcache`：存放指定 `gpu_layers` 中各层的 KV；
- `cpu_kvcache`：存放其他层的 KV。

在 decode 阶段，`SCXLlamaAttention` 会根据当前层编号和开始 / 结束标记，选择读写 `gpu_kvcache` 或 `cpu_kvcache`。

---

## 基础使用示例

下面示例展示如何在不依赖 vLLM 的情况下，直接基于 `transformers` 与 `torch` 使用 SCX 标准实现。

```python
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

# 5. 拆分 KV Cache 到 GPU 和 CPU
gpu_kvcache, cpu_kvcache = split_kvcache_dynamic(kvcache, gpu_layers=[1])

# 6. Decode 阶段
for step in range(decode_steps):
    next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    past_seen_tokens = seq_len + step
    cache_position = torch.arange(past_seen_tokens, past_seen_tokens + 1, device=device)

    output = model(
        input_ids=next_token_id,
        use_cache=True,
        mode="decode",
        cache_position=cache_position,
        gpu_kvcache=gpu_kvcache,
        cpu_kvcache=cpu_kvcache,
    )
    logits = output.logits
```

---

## 通信 / 执行框架

### 1. 组件关系

- **用户脚本**（如 `tests/llama.test.py`）：
  - 构建原始 `LlamaForCausalLM`；
  - 初始化 `SCXKeyGenerator`；
  - 调用 `encode_llama(model, key_gen)` 将 SCX 注入模型；
  - 在推理时：
    - `model(input_ids, mode="prefill")` 完成 prefill；
    - `split_kvcache_dynamic` 拆分 KV Cache；
    - `model(..., mode="decode", gpu_kvcache=..., cpu_kvcache=...)` 完成 decode。

- **模型内部结构**：
  - `LlamaForCausalLM`
    - `model.layers[i]` → 被替换为 `SCXLlamaDecoderLayer`
      - 内部 `self_attn` → 被替换为 `SCXLlamaAttention`
      - 持有当前层对应的 SCX 密钥（由 `SCXKeyGenerator.gen_keys(i)` 提供）。

### 2. Prefill 阶段数据流

1. `input_ids` 在 GPU 上经过 embedding 后进入第 0 层 `SCXLlamaDecoderLayer`；
2. 在 `SCXLlamaAttention` 中：
   - `hidden_states` GPU → CPU；
   - 可选拼接 `r_embeds`，按 `attn_pi_left` 打乱序列；
   - 回到 GPU 做 Q/K/V 投影；
   - 再到 CPU，用 `attn_inv_pi_left` 恢复原始序列顺序；
   - 应用 RoPE，并通过 `DynamicCache.update` 写入 KV Cache；
   - 在 hidden 维度按 `attn_pi_right` 打乱后回到 GPU 做注意力；
   - 注意力输出再次迁移到 CPU，通过 `attn_inv_pi_right` 与 `wo_pi_left` / `wo_inv_pi_left` 进行还原；
   - 最终返回 GPU，继续 FFN 与后续层计算。

所有层依次执行后，得到 logits 和一个包含全部层 KV 的 `DynamicCache`。

### 3. KV Cache 拆分与设备布局

1. Prefill 结束后，调用 `split_kvcache_dynamic(kvcache, gpu_layers=[...])`；
2. 函数内部遍历每层 KV，将指定 `gpu_layers` 中的层放入 `gpu_kvcache`，其余放入 `cpu_kvcache`；
3. 之后 decode 阶段，模型不再使用原先的 `past_key_values`，而是根据层类型选择从 `gpu_kvcache` 或 `cpu_kvcache` 读取 / 更新。

### 4. Decode 阶段数据流

1. 每一步 decode 输入一个 `next_token_id`，在 GPU 上 embedding 后进入各层；
2. 对于 **开始层 / 结束层**：
   - 在 CPU 上加噪 `dec_attn_alp`，并使用 `dec_attn_pi_right` / `dec_attn_inv_pi_right` 对 hidden 维度进行置换 / 还原；
   - 使用 **CPU 端 KV Cache** 进行读写；
3. 对于 **中间层**：
   - 直接使用 **GPU 端 KV Cache**，以获得更高性能；
4. 整个解码过程保持对外接口与普通 Llama 兼容，但中间状态已经被编码和跨设备分布。

---

如果你希望在论文或文档中展示更形式化的架构图（例如时序图、模块图），可以在此基础上再提炼为图形化表示。当前说明主要面向阅读本仓库代码的开发者，帮助快速理解 `scx/` 部分的结构和通信路径。


