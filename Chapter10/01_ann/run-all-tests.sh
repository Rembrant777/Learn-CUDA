#!/bin/sh 

# export cuda library for test_layer 
export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64

./test_layer    > test_layer.log
./test_helper   > test_helper.log
./test_blob     > test_blog.log
./test_loss     > test_loss.log 
./test_network  > test_network.log
./test_mnist    > test_mnist.log 