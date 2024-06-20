# `cudnnActivationDescriptor`

This enumerated type is deprecated and is currently only used by deprecated APIs.
Consider using replacements for the deprecated APIs that use this enumerated type. 

It is a pointer to an opaque structure holding the description of an activation operation.
`cudnnCreateActivationDescriptor()` is used to create one instance, and `cudnnSetActivationDescriptor()` must be used to initialize this instance.  


`cudnnActivationDescriptor_t` is the variable that can determine the types of the activation function. 
And let's review the concept of the activation function, it is the function that mapping the layer's output values into the expected output values' range. Also, it is a non-linear function, `Sigmoid`, `Relu`, `Tanh`, `Clipped Relu` and `Elu` are most frequently used activation functions. Those kind of function types are all supported to be defined in the `cudnnActivationDescriptor_t` this variable.  







## Reference
* [API Doc](https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-ops-library.html#cudnnactivationdescriptor-t)