#!/bin/sh
export CUDA_PATH=/usr/local/cuda-9.2
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${CUDA_PATH}/extras/CUPTI/lib64:${CUDA_PATH}/nvvm/lib64/:$LD_LIBRARY_PATH

python numba_matmul.py