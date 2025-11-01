import os 
from dataclasses import dataclass
from typing import Tuple

def scx_env_init(
    SCX_ENC_LAYERS: str = "", 
    SCX_SECURE_DEVICE: str = "cpu",
    SCX_ENABLE_DEBUG: bool = False):
    """
    initialize the scx os enviroments
    """
    os.environ["SCX_ENC_LAYERS"] = SCX_ENC_LAYERS
    os.environ["SCX_SECURE_DEVICE"] = SCX_SECURE_DEVICE
    os.environ["SCX_ENABLE_DEBUG"] = "True" if SCX_ENABLE_DEBUG else "False"


@dataclass(frozen=True)
class SCXConfig:
    enc_layers: Tuple[int, ...]
    secure_device: str = "cpu"
    enable_debug: bool = False


def get_scx_config() -> SCXConfig:
    # scx os env
    if os.getenv("SCX_ENC_LAYERS"):
        SCX_ENC_LAYERS = tuple([int(l) for l in os.getenv("SCX_ENC_LAYERS").split(",")])
    else:
        SCX_ENC_LAYERS = tuple()

    SCX_SECURE_DEVICE = os.getenv("SCX_SECURE_DEVICE", "cpu")

    SCX_ENABLE_DEBUG = os.getenv("SCX_ENABLE_DEBUG", "False").lower() == "true"
    
    return SCXConfig(
        enc_layers=SCX_ENC_LAYERS,
        secure_device=SCX_SECURE_DEVICE,
        enable_debug=SCX_ENABLE_DEBUG,
    )