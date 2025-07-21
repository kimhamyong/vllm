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
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import shutil
import ray

logger = init_logger(__name__)
ENABLE_LOAD_LOG = True

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
#-----------------------------------------------------------------------------------
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
    @staticmethod
    @ray.remote(num_cpus=0)
    def _pull_files(dir_root: str, tag: str, pattern: str):
        import glob, os, socket

        ip   = socket.gethostbyname(socket.gethostname())   # 실행 노드 IP
        patt = os.path.join(dir_root, pattern.format(rank=tag, part="*"))
        files = [(os.path.basename(fp), open(fp, "rb").read())
                for fp in glob.glob(patt)]
        if files:
            print(f"✅[Ray {ip}] {len(files)} file(s) matched {patt}")
        else:
            print(f"❌[Ray {ip}] no file for {patt}")
        return {"ip": ip, "files": files}

#-----------------------------------------------------------------------------------
    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.distributed import get_tensor_model_parallel_world_size
        import torch
        import ray, tempfile, os, glob, shutil
        from safetensors.torch import safe_open

        print("[👌] CustomLoader loading")

        model_weights = getattr(model_config, "model_weights", model_config.model)
        local_model_path = model_weights

        # 현재 TP rank 및 world size
        rank       = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        half       = world_size // 2

        # ── rank별로 가져올 shard 태그 결정 ─────────────────────────────
        if rank < half:               # rank 0~(half-1)
            desired_tags = (
                f"{rank}0",           # 자기 0번
                f"{rank}1",           # 자기 1번
                f"{rank+half}0",      # 뒷 노드의 0번
            )
        else:                         # rank ≥ half
            desired_tags = (f"{rank}1",)  # 자기 1번만
        print(f"🅾️[Rank {rank}] Desired tags: {desired_tags}")

        # 로컬에서 shard 파일 탐색
        filepaths, missing_tags = [], []
        for tag in desired_tags:
            pattern = os.path.join(local_model_path,
                                self.pattern.format(rank=tag, part="*"))
            found = glob.glob(pattern)
            print(f"🔵[Rank {rank}] Local Tag {tag} found files: {found}")
            if found:
                filepaths += found
                print(f"🅰️[Rank {rank}] Tag {tag} total files: {len(filepaths)}")
            else:
                missing_tags.append(tag)
                print(f"🅱️[Rank {rank}] missing_tags {tag}")

        # 부족한 태그가 있으면 원격 노드에서 수집
        if missing_tags:
            pulled = []
            for tag in missing_tags:
                print(f"😊[Rank {rank}] Searching tag {tag} on every node")
                futures = [
                    CustomLoader._pull_files.options(
                        placement_group=None,
                        num_cpus=0,
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            node_id=n["NodeID"], soft=True),
                    ).remote(local_model_path, tag, self.pattern)
                    for n in ray.nodes()
                ]
                # 모든 노드에서 결과 취합
                for res in ray.get(futures):
                    pulled.extend(res["files"])

            # pulled 파일을 임시 디렉터리로 저장
            if pulled:
                tmp_dir = tempfile.mkdtemp(prefix=f"remote_ckpt_rank{rank}_")
                print(f"✅[Rank {rank}] Saving pulled files to tmp_dir={tmp_dir}")
                for name, raw in pulled:
                    path = os.path.join(tmp_dir, name)
                    if path in filepaths:          # 중복 방지
                        continue
                    with open(path, "wb") as f:
                        f.write(raw)
                    filepaths.append(path)

        def _collect_available_keys(paths):
            keys = set()
            for fp in paths:
                with safe_open(fp, framework="pt", device="cpu") as f:
                    keys |= set(f.keys())
            return keys
        available_keys = _collect_available_keys(filepaths)

        if not filepaths:
            raise ValueError(f"❌[Rank {rank}] No shard files found")


        if not filepaths:
            # TODO: support un-sharded checkpoints too
            raise ValueError(
                f"Could not find checkpoint files '{pattern}', only "
                f"pre-sharded checkpoints are currently supported!")

        state_dict = {
            k: v
            for k, v in self._filter_subtensors(model.state_dict()).items()
            if k in available_keys          # part-0 쪽 key만 유지
        }

        # 모델 총 파라미터 수 계산
        total_params = 0

        temp_parts    = {} # half-shard buffer

        # shard 파일 순회하며 로드
        for key, tensor in self.iterate_over_files(filepaths, rank):

            loaded_params = tensor.numel()
            total_params += loaded_params

            # state_dict에 키가 없으면 스킵    
            if key not in state_dict:
                continue

            # 두 파일을 합쳐서 로드 -> 나눠진 shard 파일을 합치는 경우
            # `lm_head.weight`는 두 파일에 동일하게 저장되어 있으므로 첫 번째 텐서만 사용
            if tensor.shape != state_dict[key].shape and key != "lm_head.weight": # 텐서의 shape이 `state_dict`에 이미 저장된 파라미터와 다르면
                buf = temp_parts.setdefault(key, tensor)
                if buf is tensor: # 첫 번째 shard가 이미 저장되어 `key`가 `temp_parts`에 이미 존재하면 다음 shard가 올 때까지 기다림
                    continue
                tensor = torch.cat([buf, tensor], dim=-1) # `torch.cat()`을 사용하여 두 텐서를 합침
                temp_parts.pop(key) # 두 개의 shard가 합쳐지면, `temp_parts`에서 해당 `key` 삭제

            # tensor → param 복사 
            dst = state_dict[key].data
            for dim, size in enumerate(tensor.shape):
                if size < dst.shape[dim]:
                    dst = dst.narrow(dim, 0, size)
            dst.copy_(tensor)

            state_dict.pop(key)

        # 로딩 로그
        logger.info(f"✔️✔️[Rank {rank}] Total Loaded Parameters Across All Files: {total_params:,}")

        if state_dict:   # 남은 key = part-1 영역
            logger.info(
                "🔵[Rank %d] %d keys skipped (partial-load TP): %s",
                rank, len(state_dict), list(state_dict)[:5]
            )

#-----------------------------------------------------------------------------------
    def iterate_over_files(
            self, paths, rank) -> Generator[tuple[str, torch.Tensor], None, None]:
        if self.runai_model_streamer:
            yield from runai_safetensors_weights_iterator(paths, True)
        else:
            from safetensors.torch import safe_open

            for path in paths:
                logger.info(f"☑️[Rank {rank}] Trying to open file: {path}")

                total_file_params = 0
                with safe_open(path, framework="pt") as f:

                    for key in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(key)
                        total_file_params += tensor.numel()

                        yield key, tensor

                logger.info(f"✔️[Rank {rank}] Loaded {total_file_params:,} parameters from file {path}")

#-----------------------------------------------------------------------------------
# 기존 코드:
# 현재 rank가 가진 weight(state_dict)만 가져옴
# 각 텐서 크기를 확인해서 max_size 기준으로 분할
# 분할된 파일들을 {rank}-{part} 형태로 safetensors로 저장

# 변경 코드:
# state_dict의 각 key(파라미터)별 텐서를 1/2로 나누어
#     └─ 첫 절반 → rankX0, 두 번째 절반 → rankX1 파일에 저장


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
        import torch

        if pattern is None:
            pattern = CustomLoader.DEFAULT_PATTERN

        rank = get_tensor_model_parallel_rank() # 현재 실행 중인 rank, 저장 파일에 이 rank 값이 사용됨 -> 파일 분리 기준임

        # 저장 반복 구조 초기화
        part_idx = 0 # 하나의 rank가 저장하는 파일 번호
        total_size = 0 # 현재 파일에 누적된 파라미터 크기

        # state_dict()로 모델 파라미터를 가져옴 -> CustomLoader._filter_subtensors()로 필터링: 불필요한 파라미터 제외
        state_dict = CustomLoader._filter_subtensors(model.state_dict())

        # 저장 반복 구조 초기화: 현재 파일에 저장할 key-value 텐서 딕셔너리 (2개로 분할)
        state_dict_part_0: dict[str, torch.Tensor] = {}  # 첫 번째 절반
        state_dict_part_1: dict[str, torch.Tensor] = {}  # 두 번째 절반
        print("[👌👌] CustomLoader save_model")

        # state_dict 순회하면서 파일 분할 저장
        for key, tensor in state_dict.items():

            if key == "lm_head.weight":
                # 두 파일(0·1) 모두에 원본 그대로 복사
                state_dict_part_0[key] = tensor.contiguous()
                state_dict_part_1[key] = tensor.contiguous()
                param_size = tensor.nelement() * tensor.element_size()
                total_size += param_size
                continue

                if not logged_lm_head:          # 한 rank에서 1번만 출력
                    print(f"[CHECK][rank {rank}] lm_head.weight duplicated "
                        f"→ will be in {rank}0 / {rank}1")
                    logged_lm_head = True
                continue

            # 각 텐서를 2등분
            if len(tensor.shape) >= 1 and tensor.shape[-1] >= 2:
                # torch.split을 사용하여 contiguous한 텐서 생성
                split_size = tensor.shape[-1] // 2
                remaining_size = tensor.shape[-1] - split_size
            
                # 첫 번째 절반과 두 번째 절반으로 분할
                first_half, second_half = torch.split(tensor, [split_size, remaining_size], dim=-1)
                first_half = first_half.contiguous()
                second_half = second_half.contiguous()
            
                # 텐서가 몇 바이트를 차지하는지 계산
                param_size_0 = first_half.nelement() * first_half.element_size()
                param_size_1 = second_half.nelement() * second_half.element_size()
            
                # 저장할 텐서를 추가했을 때, 설정된 최대 파일 크기를 초과하는지 확인
                if max_size is not None and total_size + param_size_0 > max_size: # total_size: 누적 크기
                
                    # 저장할 파일 이름을 생성
                    filename_0 = pattern.format(rank=f"{rank}0", part=part_idx)
                    filename_1 = pattern.format(rank=f"{rank}1", part=part_idx)
                
                    # 딕셔너리 {key: tensor}를 .safetensors 파일로 저장
                    save_file( # 지금까지 모아둔 파라미터 묶음 (첫 번째 절반)
                        {k: v.contiguous() for k, v in state_dict_part_0.items()}, # contiguous로 변환
                        os.path.join(path, filename_0),
                    )
                    save_file( # 지금까지 모아둔 파라미터 묶음 (두 번째 절반)
                        {k: v.contiguous() for k, v in state_dict_part_1.items()}, # contiguous로 변환
                        os.path.join(path, filename_1),
                    )
                    print(f"[👌👌👌] {rank}, {filename_0}")
                    print(f"[👌👌👌] {rank}, {filename_1}")

                    # 다음에 저장할 파일의 part 번호를 1 증가 -> 하나의 rank가 여러 파일을 저장
                    part_idx += 1
                    # 누적 크기를 초기화해서 다음 파일에 텐서들을 다시 채우기 시작
                    total_size = 0
                    # save_file로 저장한 텐서 딕셔너리를 초기화
                    state_dict_part_0 = {}
                    state_dict_part_1 = {}

                # 현재 순회 중인 텐서의 절반들을 각각 state_dict_part에 추가
                state_dict_part_0[key] = first_half
                state_dict_part_1[key] = second_half
                # 이 텐서의 바이트 수를 현재 파일에 누적된 용량에 더함
                total_size += param_size_0
            
            else:
                # 분할할 수 없는 텐서는 복제
                tensor_contiguous = tensor.contiguous()
                param_size = tensor_contiguous.nelement() * tensor_contiguous.element_size()
            
                # 저장할 텐서를 추가했을 때, 설정된 최대 파일 크기를 초과하는지 확인
                if max_size is not None and total_size + param_size > max_size: # total_size: 누적 크기
                
                    # 저장할 파일 이름을 생성
                    filename_0 = pattern.format(rank=f"{rank}0", part=part_idx)
                    filename_1 = pattern.format(rank=f"{rank}1", part=part_idx)
                
                    # 딕셔너리 {key: tensor}를 .safetensors 파일로 저장
                    save_file(
                        state_dict_part_0, # 지금까지 모아둔 파라미터 묶음 (첫 번째 절반)
                        os.path.join(path, filename_0),
                    )
                    save_file(
                        state_dict_part_1, # 지금까지 모아둔 파라미터 묶음 (두 번째 절반)
                        os.path.join(path, filename_1),
                    )
                    print(f"[👌👌👌] {rank}, {filename_0}")
                    print(f"[👌👌👌] {rank}, {filename_1}")

                    # 다음에 저장할 파일의 part 번호를 1 증가 -> 하나의 rank가 여러 파일을 저장
                    part_idx += 1
                    # 누적 크기를 초기화해서 다음 파일에 텐서들을 다시 채우기 시작
                    total_size = 0
                    # save_file로 저장한 텐서 딕셔너리를 초기화
                    state_dict_part_0 = {}
                    state_dict_part_1 = {}

                # 현재 순회 중인 텐서를 양쪽 모두에 추가 (복제)
                state_dict_part_0[key] = tensor_contiguous
                state_dict_part_1[key] = tensor_contiguous
                # 이 텐서의 바이트 수를 현재 파일에 누적된 용량에 더함
                total_size += param_size

        # 반복이 끝난 후, 저장되지 않은 마지막 묶음을 저장   
        if len(state_dict_part_0) > 0:
            filename_0 = pattern.format(rank=f"{rank}0", part=part_idx)
            filename_1 = pattern.format(rank=f"{rank}1", part=part_idx)
            save_file(
                state_dict_part_0,
                os.path.join(path, filename_0),
            )
            save_file(
                state_dict_part_1,
                os.path.join(path, filename_1),
            )
            print(f"[👌👌👌✅] {rank}, {filename_0}")
            print(f"[👌👌👌✅] {rank}, {filename_1}")
