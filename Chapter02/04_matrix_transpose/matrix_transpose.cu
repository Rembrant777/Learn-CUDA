#include<stdio.h>
#include<stdlib.h>

// #define N 2048
// #define BLOCK_SIZE 32
#define N 4 
#define BLOCK_SIZE 2

__global__
void matrix_transpose_naive(int *input, int *output) 
{
    int indexX = threadIdx.x + blockIdx.x * blockDim.x; 
    int indexY = threadIdx.y + blockIdx.y * blockDim.y; 
    int index = indexY * N + indexX; 
    int transposedIndex = indexX * N + indexY; 

    output[transposedIndex] = input[index]; 
}

__global__
void matrix_transpose_shared(int *input, int *output) 
{
    __shared__ int sharedMemory[BLOCK_SIZE][BLOCK_SIZE]; 

    // global index 
    int indexX = threadIdx.x + blockIdx.x * blockDim.x; 
    int indexY = threadIdx.y + blockIdx.y * blockDim.y; 

    // transposed global memory index
    int tindexX = threadIdx.x + blockIdx.y * blockDim.x; 
    int tindexY = threadIdx.y + blockIdx.x * blockDim.y; 

    
    int index = indexY * N + indexX; 
    int transposedIndex = tindexY * N + tindexX; 


    // local index 
    int localIndexX = threadIdx.x; 
    int localIndexY = threadIdx.y; 

    // reading from global memory 
    sharedMemory[localIndexX][localIndexY] = input[index]; 

    // synchronized 
    __syncthreads(); 

    // write local shared memory value back to global memory 
    output[transposedIndex] = sharedMemory[localIndexY][localIndexX];
}

void fill_array(int *data)
{
    for (int idx = 0; idx < (N * N); idx++) {
        data[idx] = idx; 
    }
}

void print_output(int *a, int *b) 
{
    printf("\n original matrix: \n"); 
    for (int idx = 0; idx < (N * N); idx++) {
        if (idx%N == 0) {
            printf("\n"); 
        }
        printf(" %d ", a[idx]); 
    }

    for (int idx = 0; idx < (N * N); idx++) {
        if (idx%N == 0) {
            printf("\n"); 
        }
        printf(" %d ", b[idx]); 
    }
}

int main(void) 
{
    int *a, *b; 
    int *d_a, *d_b; 
    int size = N * N * sizeof(int); 

    a = (int *) malloc(size); fill_array(a); 
    b = (int *) malloc(size); 

    cudaMalloc((void **)&d_a, size); 
    cudaMalloc((void **)&d_b, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1); 
    dim3 gridSize(N / BLOCK_SIZE, N / BLOCK_SIZE, 1); 

    matrix_transpose_naive<<<gridSize, blockSize>>>(d_a, d_b);
    matrix_transpose_shared<<<gridSize, blockSize>>>(d_a, d_b);  

    cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost); 
    print_output(a, b);

    free(a); free(b); 
    
    cudaFree(d_a);  
    cudaFree(d_b);

    return 0;  
}