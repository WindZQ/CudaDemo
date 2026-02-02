#include "freshman.h"

#define BDIM 64
#define SEGM 4
#define SHFL_WIDTH 32

__global__ void test_shfl_broadcast(int* in, int* out, int const src_lane)
{
    unsigned m = __activemask();
    int value = in[threadIdx.x];
    value = __shfl_sync(m, value, src_lane, SHFL_WIDTH);
    out[threadIdx.x] = value;
}

__global__ void test_shfl_up(int* in, int* out, int const delta)
{
    unsigned m = __activemask();
    int value = in[threadIdx.x];
    value = __shfl_up_sync(m, value, delta, SHFL_WIDTH);
    out[threadIdx.x] = value;
}

__global__ void test_shfl_down(int* in, int* out, int const delta)
{
    unsigned m = __activemask();
    int value = in[threadIdx.x];
    value = __shfl_down_sync(m, value, delta, SHFL_WIDTH);
    out[threadIdx.x] = value;
}

__global__ void test_shfl_wrap(int* in, int* out, int const offset)
{
    unsigned m = __activemask();
    int value = in[threadIdx.x];

    int lane = threadIdx.x & (SHFL_WIDTH - 1);                
    int src = (lane + offset) & (SHFL_WIDTH - 1);            
    value = __shfl_sync(m, value, src, SHFL_WIDTH);

    out[threadIdx.x] = value;
}

__global__ void test_shfl_xor(int* in, int* out, int const laneMask)
{
    unsigned m = __activemask();
    int value = in[threadIdx.x];
    value = __shfl_xor_sync(m, value, laneMask, SHFL_WIDTH);
    out[threadIdx.x] = value;
}

__global__ void test_shfl_xor_array(int* in, int* out, int const laneMask)
{
    unsigned m = __activemask();

    int idx = threadIdx.x;          
    int base = idx * SEGM;           

    int value[SEGM];
#pragma unroll
    for (int i = 0; i < SEGM; ++i)
        value[i] = in[base + i];

    value[0] = __shfl_xor_sync(m, value[0], laneMask, SHFL_WIDTH);
    value[1] = __shfl_xor_sync(m, value[1], laneMask, SHFL_WIDTH);
    value[2] = __shfl_xor_sync(m, value[2], laneMask, SHFL_WIDTH);
    value[3] = __shfl_xor_sync(m, value[3], laneMask, SHFL_WIDTH);

#pragma unroll
    for (int i = 0; i < SEGM; i++)
        out[base + i] = value[i];
}

__forceinline__ __device__
void swap_seg(int* value, int lane_idx, int laneMask, int first_idx, int second_idx)
{
    unsigned m = __activemask();

    bool pred = ((lane_idx & 1) == 0);
    if (pred) {
        int tmp = value[first_idx];
        value[first_idx] = value[second_idx];
        value[second_idx] = tmp;
    }

    value[second_idx] = __shfl_xor_sync(m, value[second_idx], laneMask, SHFL_WIDTH);

    if (pred) {
        int tmp = value[first_idx];
        value[first_idx] = value[second_idx];
        value[second_idx] = tmp;
    }
}

__global__ void test_shfl_swap(int* in, int* out, int laneMask, int first_idx, int second_idx)
{
    int idx = threadIdx.x;
    int base = idx * SEGM;

    int value[SEGM];
#pragma unroll
    for (int i = 0; i < SEGM; ++i)
        value[i] = in[base + i];

    swap_seg(value, threadIdx.x, laneMask, first_idx, second_idx);

#pragma unroll
    for (int i = 0; i < SEGM; ++i)
        out[base + i] = value[i];
}

int main(int argc, char** argv)
{
    printf("starting...\n");
    init_device(0);

    int dimx = BDIM;
    unsigned int data_size = BDIM;
    int n_bytes = (int)data_size * (int)sizeof(int);

    int kernel_num = 0;
    if (argc >= 2) kernel_num = atoi(argv[1]);

    int* in_host = (int*)malloc(n_bytes);
    for (unsigned i = 0; i < data_size; ++i) in_host[i] = (int)i;

    int* out_gpu = (int*)malloc(n_bytes);

    int* in_dev = NULL;
    int* out_dev = NULL;

    CHECK(cudaMalloc((void**)&in_dev, n_bytes));
    CHECK(cudaMalloc((void**)&out_dev, n_bytes));
    CHECK(cudaMemcpy(in_dev, in_host, n_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(out_dev, 0, n_bytes));

    dim3 block(dimx);
    dim3 grid((data_size - 1) / block.x + 1);

    switch (kernel_num)
    {
    case 0:
        test_shfl_broadcast << <grid, block >> > (in_dev, out_dev, 2);
        printf("test_shfl_broadcast\n");
        break;
    case 1:
        test_shfl_up << <grid, block >> > (in_dev, out_dev, 2);
        printf("test_shfl_up\n");
        break;
    case 2:
        test_shfl_down << <grid, block >> > (in_dev, out_dev, 2);
        printf("test_shfl_down\n");
        break;
    case 3:
        test_shfl_wrap << <grid, block >> > (in_dev, out_dev, 2);
        printf("test_shfl_wrap\n");
        break;
    case 4:
        test_shfl_xor << <grid, block >> > (in_dev, out_dev, 1);
        printf("test_shfl_xor\n");
        break;
    case 5:
        test_shfl_xor_array << <1, block.x / SEGM >> > (in_dev, out_dev, 1);
        printf("test_shfl_xor_array\n");
        break;
    case 6:
        test_shfl_swap << <1, block.x / SEGM >> > (in_dev, out_dev, 1, 0, 3);
        printf("test_shfl_swap\n");
        break;
    default:
        break;
    }

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(out_gpu, out_dev, n_bytes, cudaMemcpyDeviceToHost));

    printf("input:\t");
    for (unsigned i = 0; i < data_size; i++) printf("%4d ", in_host[i]);
    printf("\noutput:\t");
    for (unsigned i = 0; i < data_size; i++) printf("%4d ", out_gpu[i]);
    printf("\n");

    cudaFree(in_dev);
    cudaFree(out_dev);
    free(in_host);
    free(out_gpu);
    cudaDeviceReset();

    return 0;
}