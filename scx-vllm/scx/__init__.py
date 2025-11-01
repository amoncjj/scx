from vllm import ModelRegistry

from .models.qwen2_scx import Qwen2SCXForCausalLM
from .models.llama_scx import LlamaSCXForCausalLM


def init_vllm_plugin():
    ModelRegistry.register_model("Qwen2SCXForCausalLM", Qwen2SCXForCausalLM)
    ModelRegistry.register_model("LlamaSCXForCausalLM", LlamaSCXForCausalLM)