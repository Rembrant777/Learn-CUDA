#include<stdio.h>
#include<stdlib.h>

#define N 512

void host_add(int *a, int *b, int *c) {
    for (int idx = 0; idx < N; idx++) {
        c[idx] = a[idx] + b[idx]; 
    }
}

__global__ void device_add(int *a, int *b, int *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x; 
    printf("-device_add-: [idx]=%d, [threadIdx.x]=%d, [blockIdx.x]=%d, [blockDim.x]=%d ;; ",
     index, threadIdx.x, blockIdx.x, blockDim.x); 
    c[index] = a[index] + b[index]; 
}

// basically just fills the array with index. 
void fill_array(int *data) {
    for (int idx = 0; idx < N; idx++) {
        data[idx] = idx; 
    }
}

void print_output(int *a, int *b, int *c) {
    for (int idx = 0; idx < N; idx++) {
        printf("%d + %d = %d\n", a[idx], b[idx], c[idx]); 
    }
}

int main(void) {
    int *ptr_a, *ptr_b, *ptr_c; 
    int *ref_a, *ref_b, *ref_c; 
    int threads_per_block = 0; 
    int no_of_blocks = 0; 
    int size = N * sizeof(int); 

    // allocate space for local device on cpu 
    ptr_a = (int *) malloc(size); fill_array(ptr_a); 
    ptr_b = (int *) malloc(size); fill_array(ptr_b);  
    ptr_c = (int *) malloc(size);

    // allocate space for gpu device 
    cudaMalloc((void **)&ref_a, size);  
    cudaMalloc((void **)&ref_b, size); 
    cudaMalloc((void **)&ref_c, size); 

    // copy data from local to device gpu 
    cudaMemcpy(ref_a, ptr_a, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(ref_b, ptr_b, size, cudaMemcpyHostToDevice); 

    // here we create 4 threads on one block 
    threads_per_block = 4; 

    // here we use total compute item counts divide by thread per block 
    // got numbers of blocks(how many block in total)
    // so that thread-1 operates range from [0, no_of_blocks -1]
    // thread-2 operates block range from [no_of_blocks, no_of_blocks * 2 - 1]
    // ... and so on 
    // we can take each thread as an parallel item 
    no_of_blocks = N/threads_per_block; 
    device_add<<<no_of_blocks, threads_per_block>>>(ref_a, ref_b, ref_c);

    // finally write results from device(gpu) back to cpu 
    cudaMemcpy(ptr_c, ref_c, size, cudaMemcpyDeviceToHost); 

    print_output(ptr_a, ptr_b, ptr_c);

    free(ptr_a); free(ptr_b); free(ptr_c); 
    cudaFree(ref_a); cudaFree(ref_b); cudaFree(ref_c); 

    return 0;   
}