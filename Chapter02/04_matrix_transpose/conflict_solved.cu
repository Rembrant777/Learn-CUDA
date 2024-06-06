#include<stdio.h>
#include<stdlib.h>

#define N 1024
#define BLOCK_SIZE 32

__global__ 
void matrix_transpose_naive(int *input, int *output) 
{
    int indexX = threadIdx.x + blockIdx.x * blockDim.x; 
    int indexY = threadIdx.y + blockIdx.y * blockDim.y; 
    int index = indexY * N + indexX; 
    int transposedIndex = indexX * N + indexY; 

    // this has discoalesced global memory store 
    output[transposedIndex] = input[index]; 
}

// this method will be executed by all the threads from different cuda blocks
__global__ 
void matrix_transpose_shared(int *input, int *output) 
{
    // this shared declared buffer is shared in the scope of cuda block
    // which can be accessed by all the threads in the current block 
    __shared__ int sharedMemory[BLOCK_SIZE][BLOCK_SIZE + 1]; 

    // global index 
    int indexX = threadIdx.x + blockIdx.x * blockDim.x; 
    int indexY = threadIdx.y + blockIdx.y * blockDim.y; 

    // transposed global memory index
    int tindexX = threadIdx.x + blockIdx.y * blockDim.x; 
    int tindexY = threadIdx.y + blockIdx.x * blockDim.y; 

    // local index
    int localIndexX = threadIdx.x; 
    int localIndexY = threadIdx.y; 

    int index = indexY * N + indexX; 
    int transposedIndex = tindexY * N + tindexX; 

    // reading from global memory in coalesed manner and performing transpose in shared memory
    sharedMemory[localIndexX][localIndexY] = input[index]; 

    // here we call kernel method to synchronized all the threads result 
    // then write to the block final result to global memory together
    __syncthreads(); 

    // writing into global memory in coalesed fashion via transposed data in shared memory 
    // use this method to resolve threads conflict 
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
    printf("Original Matrix::\n"); 
    for (int idx = 0; idx < (N * N); idx++) {
        if (idx % N == 0) {
            printf("\n"); 
        }
        printf(" %d ", a[idx]); 
    }
    printf("\n Transposed Matrix:\n"); 
    for (int idx = 0; idx < (N*N); idx++) {
        if (idx % N == 0) {
            printf("\n"); 
        }
        printf(" %d ", b[idx]); 
    }
}

int main(void) {
    int *a, *b; 
    int *d_a, *d_b; 

    int size = N * N * sizeof(int); 

    a = (int *) malloc(size); fill_array(a); 
    b = (int *) malloc(size);

    cudaMalloc((void **)&d_a, size);  
    cudaMalloc((void **)&d_b, size); 

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice); 

    // blockSize method apply how many threads per block on both {x dimension, y dimension}
	// apply on x dimension with BLOCK_SIZE thread 
	// apply on y dimension with BLOCK_SIZE thread 
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1); 

    // gridSize method apply how many cuda blocks on both {x dimension, y dimension}
	// apply on x dimension with N/BLOCK_SIZE cuda block(s)
	// apply on y dimension with N/BLOCK_SIZE cuda block(s)
    dim3 gridSize(N/BLOCK_SIZE, N/BLOCK_SIZE, 1); 

    matrix_transpose_naive<<<gridSize, blockSize>>>(d_a, d_b); 
    matrix_transpose_shared<<<gridSize, blockSize>>>(d_a, d_b); 

    cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost); 
    print_output(a, b); 

    // release memories
    free(a); 
    free(b); 
    cudaFree(d_a); 
    cudaFree(d_b);

    return 0; 
}