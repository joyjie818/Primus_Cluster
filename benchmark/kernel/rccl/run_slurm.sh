#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Slurm launcher for RCCL benchmarks inside a Docker container.
#
# Usage:
#   DOCKER_IMAGE=<image> sbatch run_slurm.sh
#   DOCKER_IMAGE=<image> NNODES=2 PARTITION=my-gpu sbatch run_slurm.sh
#  DOCKER_IMAGE=rocm/primus:v25.11 NNODES=2 sbatch -N2 -w smci355-ccs-aus-n01-[29,33] -p Compute-DCPT slurm.sh 
# DOCKER_IMAGE=rocm/primus:v25.11 NNODES=1 sbatch -N1 -w smci355-ccs-aus-n01-29 -p Compute-DCPT slurm.sh 
# DOCKER_IMAGE=rocm/megatron-lm:v25.6_py310 NNODES=2 sbatch -N2 -w smci355-ccs-aus-n01-[29,33] -p Compute-DCPT slurm.sh
# Environment variables (all optional except DOCKER_IMAGE):
#   DOCKER_IMAGE        Docker image to use (required)
#   NNODES              Number of nodes [default: 1]
#   PARTITION           Slurm partition [default: unset]
#   GPUS_PER_NODE       GPUs per node [default: 8]
#   MASTER_PORT         Port for torchrun rendezvous [default: 1234]
#   EXTRA_DOCKER_ARGS   Extra arguments passed to docker run
#
###############################################################################


#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=rccl-bench

NNODES="${NNODES:-${SLURM_NNODES:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-1234}"

SCRIPT_DIR=$SLURM_SUBMIT_DIR
OUTPUT_DIR="${SCRIPT_DIR}"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"

if [[ -z "${DOCKER_IMAGE:-}" ]]; then
    echo "[ERROR] DOCKER_IMAGE is not set. Export it before submitting the job."
    exit 1
fi
docker stop $(docker ps -q)

# Build sbatch overrides from env vars
SBATCH_OVERRIDES=()
if [[ -n "${PARTITION:-}" ]]; then
    SBATCH_OVERRIDES+=(-p "$PARTITION")
fi

echo "============================================"
echo " RCCL Benchmark - Slurm + Docker launcher"
echo "============================================"
echo "  DOCKER_IMAGE  : ${DOCKER_IMAGE}"
echo "  NNODES        : ${NNODES}"
echo "  GPUS_PER_NODE : ${GPUS_PER_NODE}"
echo "  MASTER_PORT   : ${MASTER_PORT}"
echo "  OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "============================================"
srun -N "${NNODES}" \
     --exclusive \
     --export=ALL \
     --ntasks-per-node=1 \
     "${SBATCH_OVERRIDES[@]}" \
     bash -c 'docker pull "${DOCKER_IMAGE}";docker stop $(docker ps -q)'

srun -N "${NNODES}" \
     --exclusive \
     --export=ALL \
     --ntasks-per-node=1 \
     "${SBATCH_OVERRIDES[@]}" \
     bash -c '

# ---- Resolve master address from Slurm node list ----
readarray -t NODE_ARRAY < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR="${NODE_ARRAY[0]}"
MASTER_ADDR="10.235.192.54"
NODE_RANK="${SLURM_NODEID}"

if [[ "$NODE_RANK" == "0" ]]; then
    echo "========== Slurm cluster info =========="
    echo "SLURM_NODELIST : ${NODE_ARRAY[*]}"
    echo "SLURM_NNODES   : ${SLURM_NNODES}"
    echo "MASTER_ADDR    : ${MASTER_ADDR}"
    echo "NODE_RANK      : ${NODE_RANK}"
    echo ""
fi

SCRIPT_DIR='"${SCRIPT_DIR}"'
OUTPUT_DIR='"${OUTPUT_DIR}"'
DOCKER_IMAGE='"${DOCKER_IMAGE}"'
NNODES='"${NNODES}"'
GPUS_PER_NODE='"${GPUS_PER_NODE}"'
MASTER_PORT='"${MASTER_PORT}"'
EXTRA_DOCKER_ARGS='"${EXTRA_DOCKER_ARGS:-}"'
rocm-smi

docker run --rm \
    --network=host \
    --ipc=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v "${SCRIPT_DIR}:${SCRIPT_DIR}" \
    -v "${HOME}:${HOME}" \
    -w "${SCRIPT_DIR}" \
    -e MASTER_ADDR="${MASTER_ADDR}" \
    -e MASTER_PORT="${MASTER_PORT}" \
    -e NNODES="${NNODES}" \
    -e NODE_RANK="${NODE_RANK}" \
    -e GPUS_PER_NODE="${GPUS_PER_NODE}" \
    ${EXTRA_DOCKER_ARGS} \
    "${DOCKER_IMAGE}" \
    bash -cx "
        # set -euo pipefail
        cd ${SCRIPT_DIR}
        ifconfig
        rocm-smi

        export TORCH_NCCL_HIGH_PRIORITY=1
        # export NCCL_CHECKS_DISABLE=1
        export NCCL_IB_HCA=benic1p1,benic2p1,benic3p1,benic4p1,benic5p1,benic6p1,benic7p1,benic8p1
        # export NCCL_IB_GID_INDEX=3
        # export NCCL_CROSS_NIC=0
        # export HSA_ENABLE_SDMA=0
        export NCCL_SOCKET_IFNAME=fenic
        # export GLOO_SOCKET_IFNAME=ens51f0
        # export CUDA_DEVICE_MAX_CONNECTIONS=1
        # export NCCL_PROTO=Simple
        # export RCCL_MSCCL_ENABLE=0

        PRIMUS_ROOT_PATH=\"${SCRIPT_DIR}/../../..\"
        MEGATRON_PATH=\"\${PRIMUS_ROOT_PATH}/third_party/Megatron-LM\"
        export PYTHONPATH=\"\${MEGATRON_PATH}:\${PYTHONPATH:-}\"

        echo \"[Node \${NODE_RANK}] Starting RCCL benchmarks...\"
        torchrun --master_addr \"\${MASTER_ADDR}\" \
                 --master_port \"\${MASTER_PORT}\" \
                 --nnodes=\"\${NNODES}\" \
                 --node_rank=\"\${NODE_RANK}\" \
                 --nproc_per_node=\"\${GPUS_PER_NODE}\" \
            ./benchmark_allreduce.py \
                --allreduce-report-csv-path ${OUTPUT_DIR}/allreduce_benchmark.csv \
                --allgather-report-csv-path ${OUTPUT_DIR}/allgather_benchmark.csv \
                --reducescatter-report-csv-path ${OUTPUT_DIR}/reducescatter_benchmark.csv

        echo \"[Node \${NODE_RANK}] RCCL benchmarks complete.\"
    "
'

echo ""
echo "============================================"
echo " All benchmarks finished."
echo " Results written to: ${OUTPUT_DIR}/"
echo "============================================"
