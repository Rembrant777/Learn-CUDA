#!/bin/sh
export CUDA_PATH=/usr/local/cuda-9.2

${CUDA_PATH}/bin/cuda-memcheck simple_sgemm > simple_sgemm.log 