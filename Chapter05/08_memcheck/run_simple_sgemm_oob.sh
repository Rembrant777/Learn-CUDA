#!/bin/sh
export CUDA_PATH=/usr/local/cuda-9.2

${CUDA_PATH}/bin/cuda-memcheck simple_sgemm_oob > simple_sgemm_oob.log 