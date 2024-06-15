# CUDA Library 
## `cublasXtSgemm`
### Diff between `cublasXtSgemm` and `cublasSgemm`
* `cublasSgemm` is function that belongs to cuda lib of `cuBLAS` that implements algorithms upon single GPU core. 
* `cublaxXtSgemm` is function that belongs to cuda lib of `cuBLASXt` that implements algorithms upon multiple GPU cores. 


Both `
```
C=α×op(A)×op(B)+β×C
```