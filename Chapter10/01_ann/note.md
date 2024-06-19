# Tensor 
```
I have a question here, tensor is the machine learning specified concept? 
Or CUDA inner defined concept ? 
What's the CUDA resource component associate with Tensor ? 
What's the relationship with the CUDA's `Block`, `Warp` or `Grid` `Threads`? 
```
## What's Tensor ? 
A tensor is a mathematical data structure that generalizes scalars, vectors, and matrices to higher dimensions.
In the context of machine learning and deep learning. 

* Scalars are 0-dimensional tensors
* Vector are 1-dimensional tensors
* Matrices are 2-dimensional tensors 
* Higher-dimensional tensors 

Tensors are fundamental in deep learning frameworks like TensorFlow, PyTorch, and others
representing inputs, outputs, and model parameters. 

### Relationship Between Tensors and CUDA 
1. Data Storage and Transfer:
2. Computation Acceleration:

### (Tensor) Relationship with Block, Threads, and Grids in CUDA 
1. Data Storage and Transfer
* Tensors can reside in either CPU memory or GPU memory. When using CUDA to 
accelerate tensor computations, the data needs to be transferred from the CPU to the GPU. 
* This transfer is managed using CUDA memory management APIs, such as `cudaMalloc` (to allocate GPU memory) 
and `cudaMemcpy` (to copy data between GPU and CPU)

2. Computation Acceleration:
* Deep learning frameworks use CUDA libraries like cuBLAS(for basic linear algebra subprograms) and cuDNN(for deep nerual networks) to accelerate tensor operaiton. 
* These libraries provide highly optimized routines for matrix multiplications, convolutions, and other operaitons, leveraging the parallel processing power of GPUs. 



# ANN Introduction  
Artificial Nerual Networks (ANNS) are computing systems inspired by the biological neural networks that constitute animal brains. 

## Baisc Concepts 
### Neuron (Node)
* Neuron is the fundamental unit of an ANN. Each neuron receives inputs,, processes them, and produces an output.
* Mathematically, for a single neuron: 
```
y = f( Acc(w[i] * x[i]) + b) and i in range of [1 ... n]
```
where `y` is the output, `w[i]` are the weights, `x[i]` are the inputs, b is the bias, and f is the activation function.

### Activation Function 
* Determines if a neruon should be activated or not.
* Common activation functions include 
```
1. sigmoid 
2. Tanh
3. ReLU(Rectified Linear Unit)
4. Softmax: Often used in the output layer of classification networks to represent probabilities. 
```

### Layers 
* A collection of neurons. Layers in an ANN typically include
```
1. Input Layer: The first layer that receives the input data. 
2. Hidden Layer: Intermediate layers that process inputs from the input layer. 
3. Output Layer: The final layer that produces the output of the network. 
```

### Weights and Biases
* Weights determine the strength of the connection between neruons. 
* Bias allows the activation function to be shifted to the left or right, which can be crucial for the learning process. 

## Basic Steps to Build an ANN 
### Initialization 
* Initialize weights and biases randomly or using specific techniques like Xavier or He initialization

### Forward Propagation 
* Compute the output of each layer staring from the input layer to the output layer. 
* For each neuron in layer l:
```
a[l] = f(W[l] * a[l-1] + b[l-1])
```

Where `a[l]` is the activation of the current layer, W[l] are the weights, a[l-1] is the activation of the previous layer, and b[l] is the bias. 

### Loss Function 
* Loss function is used to measure the difference between the network's output and the actual target values. 
* Common loss functions include 
```
1. Mean Squared Error(MSE): For regression tasks. 
2. Cross-Entropy Loss: For classification tasks
```

### Backward Propagation(Backpropagation):
* Compute the gradient of the loss function with respect to each weight using the chain rule.
* Update weights and biases using gradient descent or other optimization algorithms. 
* For weight update in gradient decent

### Training
* Iterate the forward and backward propagation steps over multiple epochs until the network learns the underlying patterns in the data.
* Split data into training and validation sets to evaluate the network's performance and avoid overfitting. 

# `CUDNN` CUDA Library Introduction 
`cuDNN` (CUDA Deep Nerual Network library) is a GPU-accelerated library for deep neural networks provided by NVIDIA.
It is designed to provide highly optimized implementation of standard deep learning routines such as forward and backward convolution, pooling, normalization, and activation layers. 

`cuDNN` aims to maximize the performance of deep learning applciations by levering and the parallel processing power of NVIDIA GPUs. 

## Key Features of cuDNN 
* High Performance: Provides highly optimzied implementations of key deep learning primitives. 
* Flexibility: Supports a variety of network architectures and operations. 
* Portability: Work across different NVIDIA GPU architectures. 

## `cuDNN` Functions 
`cuDNN` provides a variety of functions, including but not limited to:
* Convolutional forward and backward passes
* Pooling operations (max and average)
* Normalization(batch normalization, local response normalization)
* Activation functions(ReLU, sigmoid, tanh)
* Tensor transformations

## Overview of cuDNN API
The cuDNN API is designed to be used with the CUDA programming model. 
Here is a brief overview of how to use the `cuDNN` API:
### Initialization and Setup
* Initialize the cuDNN library context.
* Create and setup descriptors for tensors, convolutional layers, activation functions, etc. 

### Execution 
* Perform the desired operations using the appropriate cuDNN functions. 
* Cleanup and release resources. 

### Key cuDNN Data Structures 
* Handle: The cuDNN handle, which is required for all cuDNN functions calls.
* Descriptors: Describe the properties of tensors, filters, convolution operations, pooling operations, etc. 

### Detailed Function Example: Convolution Forward Pass 
Let's look at an example of performing a convolution forward pass using cuDNN.
#### Step-by-Step Process 
1. Initialize cuDNN 
```cuda 
cudnnHandle_t cudnn; 
cudnnCreate(&cudnn); 
```

2. Create Tensor Descriptors:
```cuda 
cudnnTensorDescriptor_t input_descriptor; 
cudnnCreateTensorDescriptor(&input_descriptor); 
cudnnSetTensor4dDescriptor(
    input_descriptor,
    CUDNN_TENSOR_NHWC,
    CUDNN_DATA_LOAT,
    batch_size, 
    channels,
    height,
    width); 

cudnnTensorDescriptor_t output_descriptor; 
cudnnCreateTensorDescriptor(&output_descriptor); 

cudnnSetTensor4dDescriptor(
    output_descriptor, 
    CUDNN_TENSOR_NHWC,
    CUDNN_DATA_FLOAT,
    batch_size,
    output_channels,
    output_height,
    output_width); 
```

3. Create Filter Descriptor
```cuda
cudnnFilterDescriptor_t filter_descriptor; 

cudnnCreateFilterDescriptor(&filter_descriptor); 

cudnnFilter4dDescriptor(
        filter_descriptor,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        output_channels,
        channels,
        filter_height,
        filter_width);         
```

4. Create Convolution Descriptor
```cuda 
cudnnConvolutionDescriptor_t conv_descriptor; 

cudnnCreateConvolutionDescriptor(&conv_descriptor); 

cudnnSetConvolution2dDescriptor(
    conv_descriptor,
    pad_height,
    pad_width,
    stride_height,
    stride_width,
    dilation_height,
    dilation_width,
    CUDNN_CROSS_CORRELATION,
    CUDNN_DATA_FLOAT);     
```

5. Choose Convolution Algorithm
```cuda 
cudnnConvolutionFwdAlgo_t conv_algo; 
cudnnGetConvolutionForwardAlgorithm(
        cudnn,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,// memory limit in bytes
        &conv_algo); 
```

6. Allocate Memory for Workspace 
```cuda 
size_t workspace_bytes = 0; 
cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_descriptor,
        filter_descriptor,
        conv_descriptor,
        output_descriptor,
        output_descriptor,
        conv_algo,
        &workspace_bytes); 

void* d_workspace = nullptr; 
cudaMalloc(&d_workspace, workspace_bytes);         
```

7. Perform Convolution Forward Pass:
```cuda 
const float alpha = 1.0f, beta = 0.0f; 

cudnnConvolutionForward(
    cudnn,
    &alpha,
    input_descriptor,
    d_input,
    filter_descriptor,
    d_filter,
    conv_descriptor,
    conv_algo,
    d_workspace, workspace_bytes,
    &beta,
    output_descriptor, d_output); 
```

8. Clean Up
```cuda 
cudnnDestroyTensorDescriptor(input_descriptor);

cudnnDestroyTensorDescriptor(output_descriptor);

cudnnDestroyFilterDescriptor(filter_descriptor); 

cudnnDestroyConvolutionDescriptor(conv_descriptor); 

cudaFree(d_workspace); 
cudnnDestroy(cudnn); 
```

## cuDNN Key Data Structures and Function Parameters
* `cudnnHandle_t`
This is a handle to the cuDNN library context. 
All cuDNN functions require this handle. 

* `cudnnTensorDescriptor_t`
Describes a tensor's format, dimensions, and data type.

Parameters
> `format`: Tensor layout (e.g., CUDNN_TENSOR_NCHW or CUDNN_TENSOR_CHWC)

> `dataType`: Data type of tensor elements (e.g., CUDNN_DATA_FLAOT)

> `n, c, h, w`: Dimensions of the tensor (batch size, channels, height, width)

* `cudnnFilterDescriptor_t`
Describes a filter's format, dimensions, and data type used in convolutional layers. 
Parameters:

>`dataType`: Data type of filter elements (e.g., CUDNN_DATA_FLOAT)
> `format`: Tensor layout for the filter (e.g., CUDNN_TENSOR_NCHW).
> `k, c, h, w`: Dimensions of the filter (number of output channels, number of input channels, height, width)

## cuDNN Iterms
### Channels 
* Input Channel: refer to the number of different types of input features or maps provided to a layer. This is often related to the depth of the input data. 

* Output Channel: refer to the number of different feature maps produced by a convolutional layer. 

### CNN_TENSOR_NCHW
This is a layout format used by `cuDNN` to specify the arrange of data in a multi-dimensional tensor, 
particular for deep learning applications. 

Like this one 
```
cudnnSetTensor4dDescriptor(input_descriptor, 
                           CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                           batch_size, channels, height, width);

cudnnSetTensor4dDescriptor(output_descriptor, 
                           CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
                           batch_size, height, width, channel);
    

CUDNN_TENSOR_{NHWC or NCHW}                                              

NHWC means 
N: batch_size which means the number of samples in the batch(N)
H: height which means the height of each feature map in the output tensor 
W: width which means width of each feature map in the output tensor 
C: channel which means the number of channels(depth) in the output tensor(C)
```


## References
* [YouTube](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
* Learn CUDA Programming 
* ChatGPT