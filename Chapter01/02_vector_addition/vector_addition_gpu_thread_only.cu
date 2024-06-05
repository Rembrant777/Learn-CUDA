#include<stdio.h>
#include<stdlib.h>

#define N 512

void host_add(int *a, int *b, int *c) {
    for (int idx = 0; idx < N; idx++) {
        c[idx] = a[idx] + b[idx]; 
    }
}

__global__ void device_add(int *a, int *b, int *c) {
    printf("-device_add-: threadIdx.x=%d,", threadIdx.x);
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x]; 
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

    int size = N * sizeof(int); 
    

    // allocate space on local cpu buffer and point by ptr_* pointers
    ptr_a = (int *) malloc(size); fill_array(ptr_a);
    ptr_b = (int *) malloc(size); fill_array(ptr_b); 
    ptr_c = (int *) malloc(size); // ptr_c buffer space do not need to init

    // allocate space on gpu device buffer and point them by ref_* pointers 
    cudaMalloc((void **) &ref_a, size); 
    cudaMalloc((void **) &ref_b, size); 
    cudaMalloc((void **) &ref_c, size); 

    // here we copy to be computed data from cpu space to gpu space
    // the 3rd parameter refers to the flag tell cudaMemcpy source and target
    cudaMemcpy(ref_a, ptr_a, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(ref_b, ptr_b, size, cudaMemcpyHostToDevice); 

    // execute add operation on gpu device 
    device_add<<<1, N>>>(ref_a, ref_b, ref_c);

    // copy result data from gpu device back to host(local or cpu side) 
    cudaMemcpy(ptr_c, ref_c, size, cudaMemcpyDeviceToHost); 

    // print data result from local device's buffer 
    print_output(ptr_a, ptr_b, ptr_c);

    // release allocated buffer space on both host/cpu side and device/gpu side 
    free(ptr_a); free(ptr_b); free(ptr_c); 

    cudaFree(ref_a); cudaFree(ref_b); cudaFree(ref_c); 

    return 0;  
}

