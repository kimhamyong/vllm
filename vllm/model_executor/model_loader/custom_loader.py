# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import collections
import glob
import os
from collections.abc import Generator
from typing import Any, Optional

import torch
from torch import nn

from vllm.config import LoadConfig, ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf, runai_safetensors_weights_iterator)
from vllm.transformers_utils.s3_utils import glob as s3_glob
from vllm.transformers_utils.utils import is_s3

logger = init_logger(__name__)


class CustomLoader(BaseModelLoader):
    """
    Custom model loader (based on ShardedStateLoader).
    """

    DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

    def __init__(self,
                 load_config: LoadConfig, # 모델 로딩에 필요한 설정 정보
                 runai_model_streamer: bool = False):
        super().__init__(load_config) # load_config를 상위 클래스에 넘겨주고 초기화 작업을 위임

        self.runai_model_streamer = runai_model_streamer
        extra_config = ({} if load_config.model_loader_extra_config is None
                        else load_config.model_loader_extra_config.copy())
        self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)
        if extra_config:
            raise ValueError(f"Unexpected extra config keys for load format "
                             f"{load_config.load_format}: "
                             f"{load_config.model_loader_extra_config.keys()}")


#-----------------------------------------------------------------------------------
    @staticmethod
    def _filter_subtensors(
        tensors: dict[str, torch.Tensor], ) -> dict[str, torch.Tensor]:
        same_storage_groups: dict[Any, list[tuple[str, torch.Tensor]]] = (
            collections.defaultdict(list))
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))

        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        result: dict[str, torch.Tensor] = {}
        for group in same_storage_groups.values():
            for k, t in group:
                a, b = t.data_ptr(), get_end_ptr(t)
                for k2, t2 in group:
                    if not t2.is_contiguous():
                        continue
                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t.is_contiguous():
                        break
                    if k2 < k:
                        break
                else:
                    result[k] = t
        return result

    def _prepare_weights(self, model_name_or_path: str,
                         revision: Optional[str]):
        if is_s3(model_name_or_path) or os.path.isdir(model_name_or_path):
            return model_name_or_path
        else:
            allow_patterns = ["*.safetensors"]
            return download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )


#-----------------------------------------------------------------------------------
    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model, model_config.revision)


#-----------------------------------------------------------------------------------
    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank

        model_weights = model_config.model
        if hasattr(model_config, "model_weights"):
            model_weights = model_config.model_weights
        local_model_path = model_weights
        rank = get_tensor_model_parallel_rank()
        pattern = os.path.join(
            local_model_path,
            self.pattern.format(rank=rank, part="*"),
        )

        filepaths = []
        if is_s3(local_model_path):
            file_pattern = f"*{self.pattern.format(rank=rank, part=' * ')}"
            filepaths = s3_glob(path=local_model_path,
                                allow_pattern=[file_pattern])
        else:
            filepaths = glob.glob(pattern)

        if not filepaths:
            raise ValueError(
                f"Could not find checkpoint files '{pattern}', only "
                f"pre-sharded checkpoints are currently supported!")

        state_dict = self._filter_subtensors(model.state_dict())
        for key, tensor in self.iterate_over_files(filepaths):
            param_data = state_dict[key].data
            param_shape = state_dict[key].shape
            for dim, size in enumerate(tensor.shape):
                if size < param_shape[dim]:
                    param_data = param_data.narrow(dim, 0, size)
            if tensor.shape != param_shape:
                logger.warning(
                    "loading tensor of shape %s into "
                    "parameter '%s' of shape %s",
                    tensor.shape,
                    key,
                    param_shape,
                )
            print("[👌] CustomLoader loading")
            param_data.copy_(tensor)
            state_dict.pop(key)

        if state_dict:
            raise ValueError(
                f"Missing keys {tuple(state_dict)} in loaded state!")

    def iterate_over_files(
            self, paths) -> Generator[tuple[str, torch.Tensor], None, None]:
        if self.runai_model_streamer:
            yield from runai_safetensors_weights_iterator(paths, True)
        else:
            from safetensors.torch import safe_open
            for path in paths:
                with safe_open(path, framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(key)
                        yield key, tensor

#-----------------------------------------------------------------------------------
# 기존 코드:
# 현재 rank가 가진 weight(state_dict)만 가져옴
# 각 텐서 크기를 확인해서 max_size 기준으로 분할
# 분할된 파일들을 {rank}-{part} 형태로 safetensors로 저장

# 변경 코드:
# 


    @staticmethod
    def save_model(
        # model 자체가 현재 rank 프로세스에서 동작 중인 TP 분할 모델
        model: torch.nn.Module, # 현재 rank의 TP 분할 모델 (필수)
        path: str, # 저장할 디렉토리 경로 (필수)
        pattern: Optional[str] = None, # 저장 파일 이름 패턴 -> 위에서 정의함
        max_size: Optional[int] = None, # 각 파일의 최대 크기(바이트), 초과 시 여러 파일로 쪼갬
    ) -> None:
        from safetensors.torch import save_file
        from vllm.distributed import get_tensor_model_parallel_rank

        if pattern is None:
            pattern = CustomLoader.DEFAULT_PATTERN

        rank = get_tensor_model_parallel_rank() # 현재 실행 중인 rank, 저장 파일에 이 rank 값이 사용됨 -> 파일 분리 기준임

        # 저장 반복 구조 초기화
        part_idx = 0 # 하나의 rank가 저장하는 파일 번호
        total_size = 0 # 현재 파일에 누적된 파라미터 크기

        # state_dict()로 모델 파라미터를 가져옴 -> CustomLoader._filter_subtensors()로 필터링: 불필요한 파라미터 제외
        state_dict = CustomLoader._filter_subtensors(model.state_dict())

        # 저장 반복 구조 초기화: 현재 파일에 저장할 key-value 텐서 딕셔너리
        state_dict_part: dict[str, torch.Tensor] = {}
        print("[👌👌] CustomLoader save_model")

        # state_dict 순회하면서 파일 분할 저장
        for key, tensor in state_dict.items():

            # 텐서가 몇 바이트를 차지하는지 계산
            param_size = tensor.nelement() * tensor.element_size() # 텐서 안에 있는 전체 원소 개수 * 텐서의 각 원소가 차지하는 바이트 수
            
            # 저장할 텐서를 추가했을 때, 설정된 최대 파일 크기를 초과하는지 확인
            if max_size is not None and total_size + param_size > max_size: # total_size: 누적 크기
                
                # 저장할 파일 이름을 생성
                filename = pattern.format(rank=rank, part=part_idx)
                
                # 딕셔너리 {key: tensor}를 .safetensors 파일로 저장
                # safetensors.torch.save_file
                save_file(
                    state_dict_part, # 지금까지 모아둔 파라미터 묶음
                    os.path.join(path, filename),
                )
                print(f"[👌👌👌] {rank}, {filename}")


                # 다음에 저장할 파일의 part 번호를 1 증가 -> 하나의 rank가 여러 파일을 저장
                part_idx += 1
                # 누적 크기를 초기화해서 다음 파일에 텐서들을 다시 채우기 시작
                total_size = 0
                # save_file로 저장한 텐서 딕셔너리를 초기화
                state_dict_part = {}

            # 현재 순회 중인 텐서를 state_dict_part에 추가
            state_dict_part[key] = tensor
            # 이 텐서의 바이트 수를 현재 파일에 누적된 용량에 더함
            total_size += param_size

        # 반복이 끝난 후, 저장되지 않은 마지막 묶음을 저장   
        if len(state_dict_part) > 0:
            filename = pattern.format(rank=rank, part=part_idx)
            save_file(
                state_dict_part,
                os.path.join(path, filename),
            )
