# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

from torch import nn

from vllm.config import LoadConfig, LoadFormat, ModelConfig, VllmConfig
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.bitsandbytes_loader import (
    BitsAndBytesModelLoader)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.gguf_loader import GGUFModelLoader
from vllm.model_executor.model_loader.runai_streamer_loader import (
    RunaiModelStreamerLoader)
from vllm.model_executor.model_loader.sharded_state_loader import (
    ShardedStateLoader)
from vllm.model_executor.model_loader.tensorizer_loader import TensorizerLoader
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name, get_model_architecture, get_model_cls)

from vllm.model_executor.model_loader.custom_loader import (
    CustomLoader)



def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    if isinstance(load_config.load_format, type):
        print("[✅] load_format is class type. Instantiating directly.")
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.DUMMY:
        print("[✅] DummyModelLoader selected")
        return DummyModelLoader(load_config)

    if load_config.load_format == LoadFormat.TENSORIZER:
        print("[✅] TensorizerLoader selected")
        return TensorizerLoader(load_config)

    if load_config.load_format == LoadFormat.SHARDED_STATE:
        print("[✅] ShardedStateLoader selected")
        return ShardedStateLoader(load_config)

    if load_config.load_format == LoadFormat.BITSANDBYTES:
        print("[✅] BitsAndBytesModelLoader selected")
        return BitsAndBytesModelLoader(load_config)

    if load_config.load_format == LoadFormat.GGUF:
        print("[✅] GGUFModelLoader selected")
        return GGUFModelLoader(load_config)

    if load_config.load_format == LoadFormat.RUNAI_STREAMER:
        print("[✅] RunaiModelStreamerLoader selected")
        return RunaiModelStreamerLoader(load_config)

    if load_config.load_format == LoadFormat.RUNAI_STREAMER_SHARDED:
        print("[✅] ShardedStateLoader (runai_model_streamer=True) selected")
        return ShardedStateLoader(load_config, runai_model_streamer=True)

    if load_config.load_format == LoadFormat.CUSTOM:
        print("[👌] CustomLoader selected")
        return CustomLoader(load_config)

    print("[✅] DefaultModelLoader selected")
    return DefaultModelLoader(load_config)



def get_model(*,
              vllm_config: VllmConfig,
              model_config: Optional[ModelConfig] = None) -> nn.Module:
    loader = get_model_loader(vllm_config.load_config)
    if model_config is None:
        model_config = vllm_config.model_config
    return loader.load_model(vllm_config=vllm_config,
                             model_config=model_config)


__all__ = [
    "get_model",
    "get_model_loader",
    "get_architecture_class_name",
    "get_model_architecture",
    "get_model_cls",
    "BaseModelLoader",
    "BitsAndBytesModelLoader",
    "GGUFModelLoader",
    "DefaultModelLoader",
    "DummyModelLoader",
    "RunaiModelStreamerLoader",
    "ShardedStateLoader",
    "TensorizerLoader",
    "CustomLoader",
]
