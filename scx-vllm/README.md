## SCX-vLLM 项目说明

SCX-vLLM 让 vLLM 在不改动调用方式的前提下，支持对指定层的“安全编码”（SCX）流程：在进入注意力计算前，将中间表示在安全设备上进行序列/通道两侧的置乱与旋转位置编码处理，再回到常规设备完成注意力与后续计算，最后在安全设备上恢复语义一致的输出，从而在推理链路中兼顾可用性与隐私增强。

- **支持模型**：Qwen2、Llama 系列（通过 `Qwen2SCXForCausalLM`、`LlamaSCXForCausalLM`）
- **无侵入集成**：通过 vLLM 插件注册点自动挂载，无需改动 vLLM 的使用方式
- **可配置粒度**：按 Layer 索引选择需要启用 SCX 的层；可指定安全设备（如 CPU）
- **随用随开**：通过环境变量一键启用或关闭，便于 A/B 对比与基准评测

---

### 目录结构

- `scx/`：SCX vLLM 插件与模型实现
  - `__init__.py`：向 vLLM 注册 `Qwen2SCXForCausalLM`、`LlamaSCXForCausalLM`
  - `keys.py`：SCX 环境变量初始化与读取（`scx_env_init` / `get_scx_config`）
  - `models/llama_scx.py`、`models/qwen2_scx.py`：在注意力前后注入 SCX 编码/解码逻辑
- `benchmark_throughput/`：吞吐基准脚本与日志
- `benchmark_acc/`：基于 lm-eval 的精度基准脚本与结果
- `data/`：生成固定长度输入的 prompt 工具与样例
- `test_qwen2.py`：最小可运行 Qwen2-1.5B 示例
- `pyproject.toml`：vLLM 插件入口配置：`[project.entry-points."vllm.general_plugins"]`

---

### 环境与安装

1) 准备 Python 与 CUDA/PyTorch（示例）

```bash
# 建议使用 Conda 管理环境
conda create -n scx python=3.10 -y
conda activate scx

# 根据你的 CUDA 版本选择合适的 PyTorch 轮子（示例为 CUDA 12.4）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

# vLLM 及相关依赖（pyproject 中已声明 vllm>=0.11.0）
pip install vllm transformers==4.57.0 accelerate==1.10.1 datasets==3.6.0
```

2) 安装本项目（作为 vLLM 插件）

```bash
# 在仓库根目录执行
pip install -e .
```

安装完成后，vLLM 会通过 entry-point 自动加载本插件中的 `init_vllm_plugin`，注册 SCX 模型。

---

### SCX 配置与启用

有两种方式配置 SCX：

- 方式 A：在 Python 内显式调用 `scx.keys.scx_env_init`（推荐在创建 vLLM 之前调用）
- 方式 B：直接设置环境变量（便于脚本/评测工具）

支持的配置项：

- `SCX_ENC_LAYERS`：启用 SCX 的层索引，逗号分隔。如 `"0"`、`"0,27"`。留空表示关闭。
- `SCX_SECURE_DEVICE`：安全设备标识（如 `cpu` 或你的安全环境设备名）。
- `SCX_ENABLE_DEBUG`：`True/False`，打印调试信息。

示例（方式 A，Python 内设置）：

```python
from scx.keys import scx_env_init

SCX_ENC_LAYERS = "0,27"    # 指定需要 SCX 的层索引
SCX_SECURE_DEVICE = "cpu"   # 演示用 CPU，可替换为你的安全设备
SCX_ENABLE_DEBUG = True

scx_env_init(SCX_ENC_LAYERS, SCX_SECURE_DEVICE, SCX_ENABLE_DEBUG)
```

示例（方式 B，环境变量设置）：

```bash
export SCX_ENC_LAYERS="0,27"
export SCX_SECURE_DEVICE="cpu"
export SCX_ENABLE_DEBUG="True"
```

---

### 与 vLLM 搭配使用

SCX 启用后，你可以像平常一样使用 vLLM：

```python
from vllm import LLM, SamplingParams

llm = LLM(
  model="/path/to/model",   # 例如 DeepSeek-R1-Distill-Qwen-1.5B/7B 或 Llama-70B
  dtype="bfloat16",
  tensor_parallel_size=1,    # 按需设置
  max_model_len=38000,       # 示例为 32k in + 4k out + 余量
  gpu_memory_utilization=0.92,
  enforce_eager=True,        # 开发/调试建议开启
)

sampling = SamplingParams(
  min_tokens=512,
  max_tokens=512,
  temperature=0.0,
  top_k=1,
  top_p=1.0,
)

outputs = llm.generate(["your prompt"], sampling)
print(outputs[0].outputs[0].text)
```

最小可运行示例可参考 `test_qwen2.py`。

---

### 示例脚本

- 吞吐评测（见 `benchmark_throughput/`）：
  - 运行方式（示例，以 Qwen2-1.5B 为例）：

    ```bash
    conda activate scx

    # baseline
    python benchmark_throughput/qwen2-1.5b.py 2>&1 | tee benchmark_throughput/1.5b_plaintext.log

    # 启用 SCX，仅输入置乱
    python benchmark_throughput/qwen2-1.5b.py --scx_enc_layers 0 \
      2>&1 | tee benchmark_throughput/1.5b_scx_input.log

    # 启用 SCX，输入与注意力后通道都置乱（举例多层）
    python benchmark_throughput/qwen2-1.5b.py --scx_enc_layers 0,27 \
      2>&1 | tee benchmark_throughput/1.5b_scx_io.log
    ```

  - 提供了 `qwen2-7b.py`、`llama-70b.py`，路径与并行度可按硬件调整。
  - 对应的 prompt 数据位于 `data/`，提供 `make_prompts.py` 与预生成的 `prompt_*.txt`。

- 精度评测（见 `benchmark_acc/`）：基于 lm-eval-harness
  - 运行 1.5B 示例：

    ```bash
    # 无 SCX
    bash benchmark_acc/run_eval_1.5b.sh lambada_openai benchmark_acc/results

    # 带 SCX（脚本中会设置 SCX_ENC_LAYERS=0,27）
    bash benchmark_acc/run_eval_1.5b.sh lambada_openai benchmark_acc/results
    ```

  - 同理提供 7B、70B 的脚本，可按需修改 `MODEL_PATH`、`tensor_parallel_size` 等。

评测结果样例如下（路径见 `benchmark_acc/results/**` 与 `benchmark_throughput/*.log`）。你可以对比 baseline 与启用 SCX 时的指标，以评估隐私增强带来的额外开销与精度影响。

---

### 工作原理（简述）

- 插件注册：在 `pyproject.toml` 中通过 `vllm.general_plugins` 声明入口，`scx.__init__.init_vllm_plugin` 会把 `Qwen2SCXForCausalLM`、`LlamaSCXForCausalLM` 注册到 vLLM 的 `ModelRegistry`。
- 配置读取：`scx.keys.get_scx_config` 从环境变量构造 `SCXConfig`（层索引、安全设备、调试开关）。
- 编码流程（以注意力层为例）：
  - 若当前层索引在 `SCX_ENC_LAYERS`：
    1) 将 `positions` 与 `hidden_states` 移至安全设备
    2) 对序列维进行置乱（`embeddings_pi_left`）并保留逆置乱索引
    3) Q/K/V 分解后在安全设备上执行旋转位置编码（复制一份 `rotary_emb` 保持缓存驻留）
    4) 在特征维分别对 Q 与 KV 做相同的分块置乱（并保留逆映射）
    5) 将置乱后的张量迁回常规设备执行注意力
    6) 注意力输出再迁回安全设备做逆置乱，最后回传到常规设备供后续层使用
  - 若未启用：走原生 vLLM 路径，不引入额外开销。

该流程确保即便截取中间表示，也难以复原原始输入的语义与结构；同时将高算力部分仍放在常规设备执行，控制延迟与成本。

---

### 常见问题（FAQ）

- Q: 为什么没有生效？
  - 请确认在创建 `LLM` 之前已调用 `scx_env_init` 或设置了环境变量；`SCX_ENC_LAYERS` 不能为空。
  - 开启 `SCX_ENABLE_DEBUG=True` 观察层索引与设备日志。

- Q: 层索引怎么选？
  - 与 vLLM 的层编号一致，范围 `[0, num_hidden_layers-1]`。可先选首尾若干层（如 `0,27`/`0,79`），在吞吐与精度之间折中。

- Q: 安全设备可以用 GPU 吗？
  - 取决于你的安全环境实现。示例用 `cpu` 便于演示与可移植性；实际可接入 TEE/自研安全加速器。

---

### 许可证

- 代码改编自 vLLM/Transformers 相应实现，遵循其开源许可证；本项目保留相应版权与声明头。
