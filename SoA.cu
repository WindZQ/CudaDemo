#include "freshman.h"

#define SIZE (1<<18)

struct NaiveStruct 
{
    float a[SIZE];
    float b[SIZE];
};

void sum_arrays(float* a, float* b, float* res, const int size)
{
    for (int i = 0; i < size; i++)
    {
        res[i] = a[i] + b[i];
    }
}

__global__ void sum_arrays_gpu(float* a, float* b, struct NaiveStruct* res, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        (res->a)[i] = a[i] + b[i];
    }
}

void check_result_struct(float* res_h, struct NaiveStruct* res_from_gpu_h, int elem)
{
    for (int i = 0; i < elem; i++) 
    {
        if (res_h[i] != (res_from_gpu_h->a)[i]) {
            printf("check fail!\n");
            exit(0);
        }
    }

    printf("result check success!\n");
}

int main(int argc, char** argv)
{
    int dev = 0;
    cudaSetDevice(dev);

    int elem = SIZE;
    printf("Vector size:%d\n", elem);
    int n_byte = sizeof(float) * elem;
    int n_byte_struct = sizeof(struct NaiveStruct);
    float* a_h = (float*)malloc(n_byte);
    float* b_h = (float*)malloc(n_byte);
    float* res_h = (float*)malloc(n_byte_struct);
    struct NaiveStruct* res_from_gpu_h = (struct NaiveStruct*)malloc(n_byte_struct);
    memset(res_h, 0, n_byte);
    memset(res_from_gpu_h, 0, n_byte);

    float* a_d, * b_d;
    struct NaiveStruct* res_d;
    CHECK(cudaMalloc((float**)&a_d, n_byte));
    CHECK(cudaMalloc((float**)&b_d, n_byte));
    CHECK(cudaMalloc((struct NaiveStruct**)&res_d, n_byte_struct));
    CHECK(cudaMemset(res_d, 0, n_byte_struct));
    initial_data(a_h, elem);
    initial_data(b_h, elem);

    CHECK(cudaMemcpy(a_d, a_h, n_byte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, n_byte, cudaMemcpyHostToDevice));

    dim3 block(1024);
    dim3 grid(elem / block.x);
    double start, elaps;
    start = cpu_second();
    sum_arrays_gpu << <grid, block >> > (a_d, b_d, res_d, elem);
    cudaDeviceSynchronize();
    elaps = cpu_second() - start;
    CHECK(cudaMemcpy(res_from_gpu_h, res_d, n_byte_struct, cudaMemcpyDeviceToHost));
    printf("Execution configuration <<<%d, %d>>> Time elapsed %f sec\n", grid.x, block.x, elaps);


    sum_arrays(a_h, b_h, res_h, elem);

    check_result_struct(res_h, res_from_gpu_h, elem);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);

    return 0;
}