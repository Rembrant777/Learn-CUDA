# ANN Notes 
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