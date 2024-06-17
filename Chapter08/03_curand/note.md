# Notes of CURAND 
## What's `CURAND`?
CURAND(CUDA Random Number Generation) is a part of NVIDIA CUDA library which mainly adopted to generate random data on GPU.

In `CURAND` a series of API(s) are provided to let developers to develop peseudo-random data effectively, this feature is very important in scientific, machine learning and graph processing areas. 

## What's features provided in `CURAND`?
### Psedudo Random Data Generation (PRNGs):
Multi-Psesudo Random Data Generators are provided like `XORWOW`, `MRG32k3a`, `Philox` and `MTGP32`.

### Quasi Random Sequence of Number(QRNGs):
`CURAND` supports quasi random sequence of numbers' gernation. E.g., `Sobol` and `Halton`.

### Multi-Distributed Random Sequence 
CURAND is capable of generating random numbers for a variety of distributions, including uniform, normal, lognormal, Poisson, and exponential distributions.

### Host and Device APIs 
`CURAND` supports both Host and Device random sequence of number genration. 

### API(s) Provided in `CURAND`
```cuda
curandCreateGenerator()  // create a random number generator 

curandDestroyGenerator() // destroy a random number generator

curandSetPseudoRandomGeneratorSeed() // set pseudo seed to the random number gernator 

curandGenerateUniform() // genrate uniform distribution number sequences

curandGenerateNormal() // gernate normal distribution number sequences
```

### Advantages of `CURAND`
* high performance 
* easy to use 
* multi-options 

