#!/bin/sh
KERNEL="sgemm_kernel_B"
METRICS="achieved_occupancy,flop_count_sp"

export NVCC_HOME=/usr/local/cuda-9.2

${NVCC_HOME}/bin/nvprof  --kernels "${KERNEL}" --metrics "${METRICS}"  ./sgemm 