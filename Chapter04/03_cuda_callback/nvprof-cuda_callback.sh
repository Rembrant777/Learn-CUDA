#!/bin/bash

# binary executable file
EXECUTABLE="./cuda_callback"

# profile output file name
OUTPUT_FILE="cuda_callback.nvvp"

# execute command nvprof and output prof log to .nvvp file  
nvprof -f -o ${OUTPUT_FILE} --cpu-thread-tracing on ${EXECUTABLE}
