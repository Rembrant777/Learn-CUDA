#!/bin/sh
export NVCC_HOME=/usr/local/cuda-9.2

${NVCC_HOME}/bin/nvprof -f -o profile-original-extra-opts.nvvp --profile-from-start off ./sgemm