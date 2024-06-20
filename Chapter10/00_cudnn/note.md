In this folder to better familiar with the `cudnn` library's inner defined classes and core ANN algorithm APIs.

We write series of both inner classes and API's unit test cases. 

---- 

## `cuDNN` inner Class Lists 
### `cudnnActivationMode_t`
* Usage: 
* [Unit Test]()

### `cudnnHandle_t`
* Usage:
* [Unit Test]()

### `cudnnActivationDescriptor_t`
* Usage:
* [Unit Test](https://github.com/Rembrant777/Learn-CUDA/blob/master/Chapter10/00_cudnn/cudnn_test_04/test/test_cudnn_activation_descriptor.cpp)

### `cudnnTensorDescriptor_t`
* Usage:
* [Unit Test]()


### ``


## `cuDNN` inner API Lists 
### `cudnnCreateActivationDescriptor(activation_desc: cudnnActivationDescriptor_t)`
* function signature
```cuda 
```

* usage:

* [unit test]() 


### `cudnnDestroyActivationDescriptor(activation_desc: cudnnActivationDescriptor_t)`
* function signature 
```cuda 
```

* usage:

* [unit test]() 


### `cudnnActivationForward(...)`
* function signature 
```cuda
cudnnStatus_t cudnnActivationForward(
    cudnnHandle_t handle, 
    cudnnActivationDescriptor_t activationDesc, 
    const void *alpha,
    const cudnnTensorDescriptor_t xDesc,
    const void *x,
    const void *beta,
    const cudnnTensorDescriptor_t yDesc,
    void *y
);
```
* usage:

* [unit test]() 

### `cudnnActivatoinBackward(...)`
* function signature 
```cuda 
cudnnStatus_t cudnnActivationBackward(
    cudnnHandle_t handle, 
    cudnnActivationDescriptor_t activationDesc,
    const void *alpha,
    const cudnnTensorDescriptor_t yDesc, 
    const void *y,
    const cudnnTensorDescriptor_t dyDesc,
    const void *dy,
    const cudnnTensorDescriptor_t xDesc,
    const void *x, 
    const void *beta,
    const cudnnTensorDescriptor_t dxDesc,
    void *dx
);
```

* usage:


* [unit test]() 