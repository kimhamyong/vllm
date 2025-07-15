# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import os
import shutil
from pathlib import Path

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

# 인자 파싱 함수(모델 경로, 출력 경로, TP 수)
def parse_args():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)

    # 결과 저장 경로(필수)
    parser.add_argument(
        "--output", "-o", required=True, type=str, help="path to output checkpoint"
    )
    # 저장할 파일 패턴
    parser.add_argument(
        "--file-pattern", type=str, help="string pattern of saved filenames"
    )
    # 파일 최대 크기
    parser.add_argument(
        "--max-file-size",
        type=str,
        default=5 * 1024**3,
        help="max size (in bytes) of each safetensors file",
    )
    return parser.parse_args()


def main(args):
    engine_args = EngineArgs.from_cli_args(args)
    if engine_args.enable_lora:
        raise ValueError("Saving with enable_lora=True is not supported!")
    model_path = engine_args.model
    if not Path(model_path).is_dir():
        raise ValueError("model path must be a local directory")
        
    # Create LLM instance from arguments
    # EngineArgs를 딕셔너리로 변환해서 LLM 인스턴스 생성
    # 여기서 vLLM 엔진이 메모리에 모델을 로딩함
    llm = LLM(**dataclasses.asdict(engine_args))
    # Prepare output directory
    Path(args.output).mkdir(exist_ok=True)
    # Dump worker states to output directory

    # Check which engine version is being used
    is_v1_engine = hasattr(llm.llm_engine, "engine_core")

    if is_v1_engine:
        # For V1 engine, we need to use engine_core.save_uneven
        print("Using V1 engine save path")
        llm.llm_engine.engine_core.save_uneven(
            path=args.output, pattern=args.file_pattern, max_size=args.max_file_size
        )
    else:
        # For V0 engine
        print("Using V0 engine save path")
        model_executor = llm.llm_engine.model_executor
        model_executor.save_uneven(
            path=args.output, pattern=args.file_pattern, max_size=args.max_file_size
        )

    # Copy metadata files to output directory
    for file in os.listdir(model_path):
        if os.path.splitext(file)[1] not in (".bin", ".pt", ".safetensors"):
            if os.path.isdir(os.path.join(model_path, file)):
                shutil.copytree(
                    os.path.join(model_path, file), os.path.join(args.output, file)
                )
            else:
                shutil.copy(os.path.join(model_path, file), args.output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
