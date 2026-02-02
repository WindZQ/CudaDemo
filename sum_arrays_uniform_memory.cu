#include "freshman.h"

void sum_arrays(float* a, float* b, float* res, const int size)
{
    for (int i = 0; i < size; i += 4)
    {
        res[i] = a[i] + b[i];
        res[i + 1] = a[i + 1] + b[i + 1];
        res[i + 2] = a[i + 2] + b[i + 2];
        res[i + 3] = a[i + 3] + b[i + 3];
    }
}

__global__ void sum_arrays_gpu(float* a, float* b, float* res, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        res[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv)
{
    init_device(0);

    int elem = 1 << 24;
    printf("Vector size:%d\n", elem);
    int n_byte = sizeof(float) * elem;
    float* res_h = (float*)malloc(n_byte);
    memset(res_h, 0, n_byte);

    float* a_d, * b_d, * res_d;
    CHECK(cudaMallocManaged((float**)&a_d, n_byte));
    CHECK(cudaMallocManaged((float**)&b_d, n_byte));
    CHECK(cudaMallocManaged((float**)&res_d, n_byte));

    initial_data(a_d, elem);
    initial_data(b_d, elem);

    dim3 block(512);
    dim3 grid((elem - 1) / block.x + 1);

    double start, elaps;
    start = cpu_second();
    sum_arrays_gpu << <grid, block >> > (a_d, b_d, res_d, elem);
    cudaDeviceSynchronize();
    elaps = cpu_second() - start;
    printf("Execution configuration <<<%d, %d>>> Time elapsed %f sec\n", grid.x, block.x, elaps);

    sum_arrays(b_d, b_d, res_h, elem);

    check_result(res_h, res_d, elem);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(res_h);

    return 0;
}