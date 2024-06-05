#include<stdio.h>
#include<stdlib.h>

#define N 512 

void host_add(int *a, int *b, int *c) {
    for (int idx = 0; idx < N; idx++) {
        c[idx] = a[idx] + b[idx]; 
    }
}

__global__ void device_add(int *a, int *b, int *c) {
    printf("_device_add_ blockIdx.x = %d,", blockIdx.x); 
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x]; 
}

// basically just fills the array with index. 
void fill_array(int *arr) {
    for (int idx = 0; idx < N; idx++) {
        arr[idx] = idx; 
    }
}

void print_output(int *a, int *b, int *c) {
    for (int idx = 0; idx < N; idx++) {
        printf("\n %d + %d = %d\n", a[idx], b[idx], c[idx]); 
    }
}

int main(void) {
    int *ptr_a, *ptr_b, *ptr_c; 
    // here create references of array of a,b,c
    int *ref_a, *ref_b, *ref_c;

    // allocate N items' space, each item space = sizeof(int)
    int size = N * sizeof(int);  

    // allocate space for host copies of array a,b,c and setup intput values 
    ptr_a = (int *) malloc(size); fill_array(ptr_a); 
    ptr_b = (int *) malloc(size); fill_array(ptr_b);
    ptr_c = (int *) malloc(size);  

    // allocat correspoinding space on cuda and pointed by previous create refernces
    // since in Makefile we already include cuda library into compile path
    // cudaMalloc a method defined inside the cuda library should be work as expected
    cudaMalloc((void **)&ref_a, size); 
    cudaMalloc((void **)&ref_b, size); 
    cudaMalloc((void **)&ref_c, size);


    // copy inputs to device 
    // from: cpu buffer ; to: gpu buffer
    cudaMemcpy(ref_a, ptr_a, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(ref_b, ptr_b, size, cudaMemcpyHostToDevice); 

    // here we execute the add operation on the gpu 
    // results are stored in the ptr_c pointed space
    device_add<<<N,1>>>(ref_a, ref_b, ref_c); 

    // after the above operation, all added results are stored inside of the space of ref_c
    // but, ref_c pointed buffer are belong gpu device
    // what we do next is copy the data result from ref_c gpu side to local cpu buffer space
    // which pointed by the ptr_c
    cudaMemcpy(ptr_c, ref_c, size, cudaMemcpyDeviceToHost); 

    // finall print result on ptr_c and free the space allocated on both gpu and cpu side 
    print_output(ptr_a, ptr_b, ptr_c);

    free(ptr_a); 
    free(ptr_b); 
    free(ptr_c); 
    cudaFree(ref_a); 
    cudaFree(ref_b); 
    cudaFree(ref_c); 

    return 0;  



}