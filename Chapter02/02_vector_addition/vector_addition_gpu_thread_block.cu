#include<stdio.h>
#include<stdlib.h>

#define N 512

void host_add(int *a, int *b, int *c) 
{
    for (int idx = 0; idx < N; idx++) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void device_add(int *a, int *b, int *c) 
{
    int index = threadIdx.x + blockIdx.x * blockDim.x; 
    c[index] = a[index] + b[index]; 
}

// basically just fills the array with index. 
void fill_array(int *data) 
{
    for (int idx = 0; idx < N; idx++) {
        data[idx] = idx; 
    }
}

void print_output(int *a, int *b, int *c) {
    for (int idx = 0; idx < N; idx++) {
        printf("%d + %d = %d \n", a[idx], b[idx], c[idx]); 
    }
}

int main(void) {
    int *a, *b, *c; 
    int *d_a, *d_b, *d_c; 
    int threads_per_block = 0; 
    int no_of_blocks = 0; 

    int size = N * sizeof(int); 

    // allocate space for host copies of a,b,c and setup input values 
    a = (int *) malloc(size); fill_array(a);
    b = (int *) malloc(size); fill_array(b);  
    c = (int *) malloc(size);

    // allocate space on device
    cudaMalloc((void **)&d_a, size);  
    cudaMalloc((void **)&d_b, size); 
    cudaMalloc((void **)&d_c, size);

    // copy data to device 
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice); 

    // cuda block : thread = N : 1
    // cuda block of gpu likes the thread pool of cpu 
    threads_per_block = 4; 
    no_of_blocks = N / threads_per_block; 

    // here we create N / threads_per_block = 512 / 4 = 128 blocks 
    // and in each block we allocate 4 threads to handle the add operation 
    // each block in charge of operating the range of arrays of a,b,c 
    // block-0:thread-0 process: a[0] + b[0] -> c[0] ... a[3] + b[3] -> c[3]
    // block-0:thread-1 process: a[4] + b[4] -> c[4] ... a[7] + b[7] -> c[7]
    // ...
    // block-127:thread-3 process: a[508] + b[508] -> c[508] ... a[511] + b[511] -> c[511]
    // all threads are executes parallel, and each block handle different fraction of the array
    // the data are isolated in different blocks
    // <<<<X,Y>>> X refers to how many blocks apply from gpu to handle computation 
    // Y refers to how how many threads apply from each block to handle computation
    device_add<<<no_of_blocks, threads_per_block>>>(d_a, d_b, d_c);

    // copy result back to host 
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);  

    print_output(a, b, c); 
    free(a); free(b); free(c); 

    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c);  

    return 0; 
}