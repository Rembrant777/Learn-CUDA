#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

float *getMatrix(const int n_row, const int n_col); 
void printMatrix(const float *matrix, const int m, const int ldm); 

int main()
{
    cublasHandle_t handle; 

    // prepare input matrics 
    float *A, *B, *C; 
    int M, N, K; 
    float alpha, beta; 

    M = 3; 
    N = 4; 
    K = 7; 
    alpha = 1.f; 
    beta = 0.f; 

    // create cuBLAS handle 
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS initializaiton failed" << std::endl; 
        return EXIT_FAILURE; 
    }

    srand(2024); 

    A = getMatrix(K, M); 
    B = getMatrix(N, K); 
    C = getMatrix(M, N); 

    std::cout << "A: " << std::endl; 
    printMatrix(A, K, M); 
    std::cout << "B: " << std::endl; 
    printMatrix(B, N, K); 
    std::cout << "C: " << std::endl; 
    printMatrix(C, M, N);

    // Gemm that trying to calculate
    // C = alpha * op(A) * op(A) + beta * op(C)
    // and here the op refers to an operation that can be applied to the matrics A and B. 
    // Specifically, op can be one of the following operations

    // 1. no operation(N): --> CUBLAS_OP_N for no operation; 
    // The matrix is used as is, without any transformation.

    // 2. Transpose(T): --> CUBLAS_OP_T for transpose; 

    // The matrix is transposed, which means rows and columns are swapped. 

    // 3. Conjugate Transpose(C):  --> CUBLAS_OP_C for conjugate transpose; 
    // For complex matrices, the matrix is transposed and then the complex conjugate of each element is taken. 




    /**
     here in this function
     alpha and beta are scalars. 
     A, B and C are matrics
     param1:    handle, A handle to the cuBLAS library context. 
                Before we call `cublasSgemm`, we must make sure that 
                `cublasCreate` function must be created first to initialize the cuBLAS library context. 
     
     param2:    CUBLAS_OP_T(opA): specifies the operation to be performed on matrix A. CUBLAS_OP_T indicates
                that matrix A should be transposed. 

     
     param3:    CUBLAS_OP_T(opB): specifies the operation to be performed on matrix B. CUBLAS_OP_T indicates 
                that matrix B should be transposed too. 

     param4:    M the number of rows in the resulting matrix C.

     param5:    N the number of cols in the resulting matrix C. 

     param6:    K the inner dimension, which is the number of columns of op(A) and the number of rows of op(B).

     param7:    &alpha A pointer points to the scalar alpha 

     param8:    A A pointer points to the matrix A on the device(GPU).

     param9:    K The leading dimension of A, which is the number of elements between successive rows
                in the memory when A is trated as a 1 D array. 
    
     param10:   B A pointer points to the matrix B on the device(GPU).

     param11:   N The leading dimension of B, which is the number of elements between successive rows
                in memory when B is treated as a 1 D array. Since B is transposed, this is the original number 
                of columns of B. 

     param12:   &beta A pointer to the scala beta. 

     param13:   C A pinter points to the matrix C on the device. 

     param14:   The leading dimension of C, which is the number of elements between successive rows in the 
                memory for matrix C.   
    */
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
        M, N, K,
        &alpha,
        A, K,
        B, N,
        &beta,
        C, M);

    cudaDeviceSynchronize(); 
    std::cout<< "C out: " << std::endl; 
    printMatrix(C, M, N); 

    cublasDestroy(handle); 

    cudaFree(A);
    cudaFree(B);  
    cudaFree(C); 

    return 0; 
}

/**
  Function to apply space on device(GPU) with initialized random data.
  param-1: m rows of to be created matrix
  param-2: ldm leading dimension 

  return created and init ok matrix 
*/
float* getMatrix(const int m, const int ldm)
{
    float *pf_matrix = nullptr; 
    cudaMallocManaged((void **)&pf_matrix, sizeof(float) * ldm * m); 

    for (int j = 0; j < m; j++) {
        for (int i = 0; i < ldm; i++) {
            // IDX2 is a macro defined calculator 
            // to map the linear value of i,j, ldm into the 2D index(i,j)
            pf_matrix[IDX2C(i, j, ldm)] = (float) rand() / RAND_MAX; 
        }
    }

    return pf_matrix; 
}


void printMatrix(const float* matrix, const int m, const int ldm)
{
    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < ldm; i++)
        {
            std::cout << std::fixed << std::setw(8) << std::setprecision(4) << matrix[IDX2C(i, j, ldm)];
        }
        std::cout << std::endl;
    }
}