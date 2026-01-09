"""
Experiment configs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ModelConfig:
    """
    Currently support only OPENAI compatible api base.
    """

    model_name: str
    api_base_env: str
    api_key_env: str


@dataclass
class QwenVLFlashConfig(ModelConfig):
    model_name: str = "qwen3-vl-flash"
    api_base_env: str = "DASHSCOPE_API_BASE"
    api_key_env: str = "DASHSCOPE_API_KEY"


@dataclass
class GeminiFlashConfig(ModelConfig):
    model_name: str = "google/gemini-2.5-flash"
    api_base_env: str = "OPENROUTER_API_BASE"
    api_key_env: str = "OPENROUTER_API_KEY"


@dataclass
class GeminiFlashLiteConfig(ModelConfig):
    model_name: str = "google/gemini-2.5-flash-lite"
    api_base_env: str = "OPENROUTER_API_BASE"
    api_key_env: str = "OPENROUTER_API_KEY"


@dataclass
class GPTConfig(ModelConfig):
    model_name: str = "openai/gpt-4o-mini"
    api_base_env: str = "OPENROUTER_API_BASE"
    api_key_env: str = "OPENROUTER_API_KEY"



@dataclass
class GemmaConfig(ModelConfig):
    model_name: str = "gemma3:latest"
    api_base_env: str = "OLLAMA_API_BASE"
    api_key_env: str = "OLLAMA_API_KEY"


@dataclass
class Qwen8bConfig(ModelConfig):
    model_name: str = "qwen3-vl:8b"
    api_base_env: str = "OLLAMA_API_BASE"
    api_key_env: str = "OLLAMA_API_KEY"


@dataclass
class Qwen32bConfig(ModelConfig):
    model_name: str = "qwen3-vl:32b-thinking"
    api_base_env: str = "OLLAMA_API_BASE"
    api_key_env: str = "OLLAMA_API_KEY"


@dataclass
class DatasetConfig:
    question_file: Path
    output_dir: Path


@dataclass
class HM3DFullConfig(DatasetConfig):
    question_file: Path = Path("data/open-eqa-hm3d-full.json")
    output_dir: Path = Path("results/oeqa-hm3d-full/")


@dataclass
class OpenEQAConfig:
    defaults: List[Any] = field(
        default_factory=lambda: [{"dataset": "full"}, {"model": "glm"}, "_self_"]
    )

    dataset: DatasetConfig = MISSING
    dataset_dir: Path = Path("data/frames")
    model: ModelConfig = MISSING
    agent: str = "baseline"  # "cov" or "baseline"
    max_views_k: int = 5
    min_action_step: int = 3


# https://dashscope.aliyuncs.com/compatible-mode/v1
# https://dashscope-intl.aliyuncs.com/compatible-mode/v1
# https://generativelanguage.googleapis.com/v1beta/openai/
# https://openrouter.ai/api/v1/
# https://open.bigmodel.cn/api/paas/v4/

cs = ConfigStore.instance()
cs.store(name="openeqa", node=OpenEQAConfig)

cs.store(group="dataset", name="full", node=HM3DFullConfig)

cs.store(group="model", name="qwen", node=QwenVLFlashConfig)
cs.store(group="model", name="gemini", node=GeminiFlashConfig)
cs.store(group="model", name="gemini_lite", node=GeminiFlashLiteConfig)
cs.store(group="model", name="gemma", node=GemmaConfig)
cs.store(group="model", name="qwen8b", node=Qwen8bConfig)
cs.store(group="model", name="qwen32b", node=Qwen32bConfig)
cs.store(group="model", name="gpt", node=GPTConfig)
