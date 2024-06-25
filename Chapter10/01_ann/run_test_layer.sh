#!/bin/sh 

# export cuda library for test_layer 
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64

./test_layer > test_layer.log
