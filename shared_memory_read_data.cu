#include "freshman.h"

#define BDIMX 32
#define BDIMY 32

#define BDIMX_RECT 32
#define BDIMY_RECT 16
#define IPAD 1

__global__ void warmup(int* out)
{
	__shared__ int tile[BDIMY][BDIMX];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();

	out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void set_row_read_row(int* out)
{
	__shared__ int tile[BDIMY][BDIMX];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();

	out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void set_col_read_col(int* out)
{
	__shared__ int tile[BDIMY][BDIMX];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();

	out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void set_col_read_row(int* out)
{
	__shared__ int tile[BDIMY][BDIMX];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();

	out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void set_row_read_col(int* out)
{
	__shared__ int tile[BDIMY][BDIMX];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();

	out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void set_row_read_col_dyn(int* out)
{
	extern __shared__ int tile[];
	unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;

	tile[row_idx] = row_idx;
    __syncthreads();

	out[row_idx] = tile[col_idx];
}

__global__ void set_row_read_col_ipad(int* out)
{
	__shared__ int tile[BDIMY][BDIMX + IPAD];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

	tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();

	out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void set_row_read_col_dyn_ipad(int* out)
{
	extern __shared__ int tile[];
	unsigned int row_idx = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
	unsigned int col_idx = threadIdx.x * (blockDim.x + 1) + threadIdx.y;

	tile[row_idx] = row_idx;
    __syncthreads();

	out[row_idx] = tile[col_idx];
}

__global__ void set_row_read_col_rect(int* out)
{
	__shared__ int tile[BDIMY_RECT][BDIMX_RECT];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int icol = idx % blockDim.y;
	unsigned int irow = idx / blockDim.y;

	tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();

	out[idx] = tile[icol][irow];
}

__global__ void set_row_read_col_rect_dyn(int* out)
{
	extern __shared__ int tile[];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int icol = idx % blockDim.y;
	unsigned int irow = idx / blockDim.y;
	unsigned int col_idx = icol * blockDim.x + irow;

	tile[idx] = idx;
    __syncthreads();

	out[idx] = tile[col_idx];
}

__global__ void set_row_read_col_rect_pad(int* out)
{
	__shared__ int tile[BDIMY_RECT][BDIMX_RECT + IPAD * 2];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int icol = idx % blockDim.y;
	unsigned int irow = idx / blockDim.y;

	tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();

	out[idx] = tile[icol][irow];
}

__global__ void set_row_read_col_rect_dyn_pad(int* out)
{
	extern __shared__ int tile[];
	unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int icol = idx % blockDim.y;
	unsigned int irow = idx / blockDim.y;
	unsigned int row_idx = threadIdx.y * (IPAD + blockDim.x) + threadIdx.x;
	unsigned int col_idx = icol * (IPAD + blockDim.x) + irow;

	tile[row_idx] = idx;
    __syncthreads();

	out[idx] = tile[col_idx];
}

int main(int argc, char** argv)
{
    init_device(0);
    int kernel = 0;
    if (argc >= 2) {
        kernel = atoi(argv[1]);
    }

    int elem = BDIMX * BDIMY;
    printf("Vector size:%d\n", elem);
    int n_byte = sizeof(int) * elem;
    int* out;
    CHECK(cudaMalloc((int**)&out, n_byte));
    cudaSharedMemConfig MemConfig;
    CHECK(cudaDeviceGetSharedMemConfig(&MemConfig));
    printf("--------------------------------------------\n");
    switch (MemConfig) 
    {
    case cudaSharedMemBankSizeFourByte:
        printf("the device is cudaSharedMemBankSizeFourByte: 4-Byte\n");
        break;
    case cudaSharedMemBankSizeEightByte:
        printf("the device is cudaSharedMemBankSizeEightByte: 8-Byte\n");
        break;

    }
    printf("--------------------------------------------\n");
    dim3 block(BDIMY, BDIMX);
    dim3 grid(1, 1);
    dim3 block_rect(BDIMX_RECT, BDIMY_RECT);
    dim3 grid_rect(1, 1);
    warmup << <grid, block >> > (out);
    printf("warmup!\n");
    double start, elaps;
    start = cpu_second();
    switch (kernel)
    {
    case 0:
    {
        set_row_read_row << <grid, block >> > (out);
        cudaDeviceSynchronize();
        elaps = cpu_second() - start;
        printf("set_row_read_row  ");
        printf("Execution Time elapsed %f sec\n", elaps);
        start = cpu_second();
        set_col_read_col << <grid, block >> > (out);
        cudaDeviceSynchronize();
        elaps = cpu_second() - start;
        printf("set_col_read_col  ");
        printf("Execution Time elapsed %f sec\n", elaps);
        break;
    }
    case 2:
    {
        set_col_read_row << <grid, block >> > (out);
        cudaDeviceSynchronize();
        elaps = cpu_second() - start;
        printf("set_col_read_row  ");
        printf("Execution Time elapsed %f sec\n", elaps);
        break;
    }
    case 3:
    {
        set_row_read_col << <grid, block >> > (out);
        cudaDeviceSynchronize();
        elaps = cpu_second() - start;
        printf("set_row_read_col  ");
        printf("Execution Time elapsed %f sec\n", elaps);
        break;
    }
    case 4:
    {
        set_row_read_col_dyn << <grid, block, (BDIMX)*BDIMY * sizeof(int) >> > (out);
        cudaDeviceSynchronize();
        elaps = cpu_second() - start;
        printf("set_row_read_col_dyn  ");
        printf("Execution Time elapsed %f sec\n", elaps);
        break;
    }
    case 5:
    {
        set_row_read_col_ipad << <grid, block >> > (out);
        cudaDeviceSynchronize();
        elaps = cpu_second() - start;
        printf("set_row_read_col_ipad  ");
        printf("Execution Time elapsed %f sec\n", elaps);
        break;
    }
    case 6:
    {
        set_row_read_col_dyn_ipad << <grid, block, (BDIMX + IPAD) * BDIMY * sizeof(int) >> > (out);
        cudaDeviceSynchronize();
        elaps = cpu_second() - start;
        printf("set_row_read_col_dyn_ipad  ");
        printf("Execution Time elapsed %f sec\n", elaps);
        break;
    }
    case 7:
    {
        set_row_read_col_rect << <grid_rect, block_rect >> > (out);
        cudaDeviceSynchronize();
        elaps = cpu_second() - start;
        printf("set_row_read_col_rect  ");
        printf("Execution Time elapsed %f sec\n", elaps);
        break;
    }
    case 8:
    {
        set_row_read_col_rect_dyn << <grid_rect, block_rect, (BDIMX) * BDIMY * sizeof(int) >> > (out);
        cudaDeviceSynchronize();
        elaps = cpu_second() - start;
        printf("set_row_read_col_rect_dyn  ");
        printf("Execution Time elapsed %f sec\n", elaps);
        break;
    }
    case 9:
    {
        set_row_read_col_rect_pad << <grid_rect, block_rect >> > (out);
        cudaDeviceSynchronize();
        elaps = cpu_second() - start;
        printf("set_row_read_col_rect_pad  ");
        printf("Execution Time elapsed %f sec\n", elaps);
        break;
    }
    case 10:
    {
        set_row_read_col_rect_dyn_pad << <grid_rect, block_rect, (BDIMX + 1) * BDIMY * sizeof(int) >> > (out);
        cudaDeviceSynchronize();
        elaps = cpu_second() - start;
        printf("set_row_read_col_rect_dyn_pad  ");
        printf("Execution Time elapsed %f sec\n", elaps);
        break;
    }
    case 11:
    {
        set_row_read_row << <grid, block >> > (out);
        cudaDeviceSynchronize();

        set_col_read_col << <grid, block >> > (out);
        cudaDeviceSynchronize();

        set_col_read_row << <grid, block >> > (out);
        cudaDeviceSynchronize();

        set_row_read_col << <grid, block >> > (out);
        cudaDeviceSynchronize();

        set_row_read_col_dyn << <grid, block, (BDIMX) * BDIMY * sizeof(int) >> > (out);
        cudaDeviceSynchronize();

        set_row_read_col_ipad << <grid, block >> > (out);
        cudaDeviceSynchronize();

        set_row_read_col_dyn_ipad << <grid, block, (BDIMX + IPAD) * BDIMY * sizeof(int) >> > (out);
        cudaDeviceSynchronize();
        break;
    }
    case 12:
    {
        set_row_read_col_rect << <grid_rect, block_rect >> > (out);
        set_row_read_col_rect_dyn << <grid_rect, block_rect, (BDIMX) * BDIMY * sizeof(int) >> > (out);
        set_row_read_col_rect_pad << <grid_rect, block_rect >> > (out);
        set_row_read_col_rect_dyn_pad << <grid_rect, block_rect, (BDIMX + 1) * BDIMY * sizeof(int) >> > (out);
        break;
    }

    }

    cudaFree(out);
    return 0;
}