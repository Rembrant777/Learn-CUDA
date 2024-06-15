# CUDA Library 
## `cublasXtSgemm`
### Diff between `cublasXtSgemm` and `cublasSgemm`
* `cublasSgemm` is function that belongs to cuda lib of `cuBLAS` that implements algorithms upon single GPU core. 
* `cublaxXtSgemm` is function that belongs to cuda lib of `cuBLASXt` that implements algorithms upon multiple GPU cores. 


* Both `cublasSgemm` and `cublasXtSgemm` are trying to calculate the function below: 
```
C=α×op(A)×op(B)+β×C
```

`op(A)` means matrix operation upon the Matrix A. 
There are three parameters represent differnt operation type of operation upon the given Matrix. 
1. `CUBLAS_OP_N` means no operations will be operated upon the Matrix. 
2. `CUBLAS_OP_T` means there are some transpose this operation executes upon the Matrix. 
3. `CUBLAS_OP_C` means conjugate transpose this operation executes upon the Matrix. 

## Parameters in Function `cublasXtSgemm`

```cpp 
// C=α×op(A)×op(B)+β×C
cublasXtSgem(handle: cublasXtHandle_t, 
            CUBLAS_OP_N: cublasOperation_t, 
            CUBLAS_OP_N: cublaxOperation_t,
            M: int, 
            N: int, 
            K: int,
            &alpha: const float*,
            A: const float*, 
            lda: int,
            B: const float*,
            ldb: int, 
            &beta: const float*, 
            C: float*, 
            ldc: int)
```
* 1. `handle: cublasXthandle_t`: this is the cublaxXt lib's context, created via function `cublasXtCreate`.
* 2. `CUBLAS_OP_N: cublasOperatoin_t`: this defines how to manipulate the 1st Matrix(A) via different flag values. 
* 3. `CUBLAS_OP_N: cublasOperation_t`: this defines how to manipulate the 2nd Matrix(B) via different flag values. 
* 4. `M: int`: rows in Matrix C and Matrix A
* 5. `N: int`: cols in Matrix C and rows in Matrix B
* 6. `K: int`: cols in Matrix A and rows in Matrix B
* 7. `alpha: const float *`: pointer of scalar alpha, used to multiply to op(A)
* 8. `A: const float*`: pointer of Matrix A
* 9. `lda: int`: Matrix A's leading dimension(row number if not transpose)
* 10. `B: const float*`: pointer of Matrix B
* 11. `ldb: int`: Matrix B's leading dimension(column number, if not transpose)
* 12. `beta: const float *`: pointer of scalar beta, use to multiply to op(C)
* 13. `C: const float*`: pointer of Matrix C
* 14. `ldc: int`: Matrix C's leading dimension(column number , if not transposed)

* Invoke Example 

```cpp
cublasXtHandle_t handle;
cublasXtCreate(&handle);

float alpha = 1.0f;
float beta = 0.0f;
int M = 640;  
int N = 480;  
int K = 160; 

float *A;  
float *B;  
float *C;  

// Here, suppose all A, B, and C are initialized
cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              M, N, K,
              &alpha, A, M,
              B, K, &beta, C, M);

cublasXtDestroy(handle);
```