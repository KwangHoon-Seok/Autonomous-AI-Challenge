#!/usr/bin/env bash

set -x
NGPUS=$1
# NGPUS=${1:-4}  # 첫 번째 인자가 없을 경우 기본값을 4로 설정 24.10.24 modified by jw
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

# origin
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS}

# 24.10.24 modified by jw
# CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nnodes=1 --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS}

