# Convolutional nerual networks with cuDNN 

The implementation of a convolutional layer is similar to the fully connected layer implementation.
Like other layers, it has three work phases. 

* For the inference phases, we will call `cudnnConvolutionForward()` and `cudnnAddTensor()` functions. 

* For the backward phase, we will call `cudnnConvolutionBackwardData()`, `cudnnConvolutionBackwardFilter()` and `cudnnConvolutionBackwardBias()`. 

* Finally, for the update phase we can reuse the code from the fully connected layers. 


To faimiliar with those `cuDNN` library APIs, we add different unit tests to test the APIs. 

## CNN Phase-1 `cudnnAddTensor`
* [Unit Test]()

* Description 
```
```

## CNN Phase-1 `cudnnConvolutionForward`
* [Unit Test]()

* Description 
```
```

## CNN Phase-2 `cudnnConvolutionBackwardData`
* [Unit Test]()

* Description 
```
```

## CNN Phase-2 `cudnnConvolutionBackwardFilter`
* [Unit Test]()

* Description 
```
```

## CNN Phase-2 `cudnnConvolutionBackwardBias`
* [Unit Test]()

* Description 
```
```