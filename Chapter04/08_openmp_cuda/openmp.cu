#include <cstdio>
#include <omp.h>
#include <helper_timer.h>

using namespace std; 

__global__ void vecAdd_kernel(float *c, const float *a, const float *b); 
void init_buffer(float *data, const int size); 

class Operator
{
private:
    int index; 
    cudaStream_t stream; 
    StopWatchInterface *p_timer; 
    static void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void *userData);     
    void print_timer(); 
public:
    Operator() {
        cudaStreamCreate(&stream); 
        sdkCreateTimer(&p_timer); 
    }    

    ~Operator() {
        cudaStreamDestroy(stream); 
        sdkDeleteTimer(&p_timer);
    }

    void set_index(int idx) {
        index = idx; 
    }

    void async_operation(float *h_c, const float *h_a, const float *h_b,
                float *d_c, float *da, float *d_b,
                const int size, const int bufsize); 
}; // Operator 

void Operator::CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void *userData)
{
    Operator* this_ = (Operator*) userData; 
    this_->print_timer(); 
}

void Operator::print_timer() 
{
    sdkStopTimer(&p_timer); 
    float elapsed_time_msed = sdkGetTimerValue(&p_timer); 
    printf("stream %2d - elapsed %.3f ms\n", index, elapsed_time_msed); 
}

void Operator::async_operation(float *h_c, const float *h_a, const float *h_b,
                    float *d_c, float *d_a, float *d_b,
                    const int size, const int bufsize)
{
    // start timer
    sdkStartTimer(&p_timer); 

    // copy host -> device 
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, stream); 
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, stream); 

    // launch cuda kernel 
    // 256 threads per block 
    dim3 dimBlock(256); 

    // size / 256 blocks per grid 
    dim3 dimGrid(size / dimBlock.x); 
    vecAdd_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_c, d_a, d_b);

    // copy device -> host 
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, stream); 

    // register callback function 
    cudaStreamAddCallback(stream, Operator::Callback, this, 0); 
}                    

int main(int argc, char *argv[])
{
    float *h_a, *h_b, *h_c; 
    float *d_a, *d_b, *d_c; 
    int size = 1 << 24; 
    int bufsize = size * sizeof(float); 
    int num_operator = 4; 

    if (argc != 1) {
        num_operator = atoi(argv[1]);
    }

    // initialize timer
    StopWatchInterface *timer; 
    sdkCreateTimer(&timer); 

    // allocate host memories 
    cudaMallocHost((void **)&h_a, bufsize); 
    cudaMallocHost((void **)&h_b, bufsize);
    cudaMallocHost((void **)&h_c, bufsize);

    // initialize host values 
    srand(2024); 
    init_buffer(h_a, size); 
    init_buffer(h_b, size); 
    init_buffer(h_c, size); 

    // allocate device memories 
    cudaMalloc((void**) &d_a, bufsize); 
    cudaMalloc((void**) &d_b, bufsize);
    cudaMalloc((void**) &d_c, bufsize);

    // create list of operation elements
    Operator* ls_operator = new Operator[num_operator]; 

    sdkStartTimer(&timer); 

    // execute each operator correspoinding data
    // thread-resource allocated from host(cpu) side
    // block below will be executed in parallel
    // parallelism = max_threads_applied / total tasks 
    // this is cpu threads parallel unit
    omp_set_num_threads(num_operator); 
    #pragma omp parallel
    {
        // get current thread idx
        int i = omp_get_thread_num(); 
        // get init offset of to be computed array 
        int offset = i * size / num_operator; 

        // first set index of current thread to operator item 
        ls_operator[i].set_index(i);

        // here invoke cuda parallel operation 
        // this is cuda parallel unit 
        ls_operator[i].async_operation(&h_c[offset], &h_a[offset], &h_b[offset],
                            &d_c[offset], &d_a[offset], &d_b[offset],
                            size / num_operator, bufsize / num_operator); 
    }

    // sync all stream operation 
    cudaDeviceSynchronize(); 
    sdkStopTimer(&timer);

    // print out the result 
    int print_idx = 256; 
    printf("compared a sample result ...\n");  
    printf("host: %.6f, device: %.6f\n", h_a[print_idx] + h_b[print_idx], h_c[print_idx]);

    // compute and print the performance 
    float elapsed_time_msed = sdkGetTimerValue(&timer); 
    float bandwidth = 3 * bufsize * sizeof(float) / elapsed_time_msed / 1e6; 
    printf("Time = %.3f msec, bandwidth = %f GB/s\n", elapsed_time_msed, bandwidth); 

    // delete timer
    sdkDeleteTimer(&timer); 

    // terminate operators 
    delete [] ls_operator; 

    // terminate device memories
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c); 

    // terminate host memories 
    cudaFreeHost(h_a); 
    cudaFreeHost(h_b); 
    cudaFreeHost(h_c); 
    
    return 0; 
}

__global__ 
void vecAdd_kernel(float *c, const float* a, const float* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < 500; i++)
        c[idx] = a[idx] + b[idx];
}

void init_buffer(float *data, const int size)
{
    for (int i = 0; i < size; i++) 
        data[i] = rand() / (float)RAND_MAX;
}

