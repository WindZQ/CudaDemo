#include "freshman.h"

#define BDIMX 8
#define BDIMY 8
#define IPAD 2

void transform_matrix2d_cpu(float* in, float* out, int nx, int ny)
{
	for (int j = 0; j < ny; ++j)
	{
		for (int i = 0; i < nx; ++i)
		{
			out[i * nx + j] = in[i + j * nx];
		}
	}
}

__global__ void warmup(float* in, float* out, int nx, int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = ix + iy * nx;

	if (ix < nx && iy < ny) {
		out[idx] = in[idx];
	}
}

__global__ void copy_row(float* in, float* out, int nx, int ny)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = ix + iy * nx;

	if (ix < nx && iy < ny) {
		out[idx] = in[idx];
	}
}

__global__ void transform_naive_row(float* in, float* out, int nx, int ny)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;
	int iy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx_row = ix + iy * nx;
	int idx_col = ix * ny + iy;

	if (ix < nx && iy < ny) {
		out[idx_col] = in[idx_row];
	}
}

__global__ void transform_smem(float* in, float* out, int nx, int ny)
{
	__shared__ float tile[BDIMY][BDIMX];
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int transform_in_idx = ix + iy * nx;

	unsigned int bidx = threadIdx.x + threadIdx.y * blockDim.x;
	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.y;

	ix = icol + blockIdx.y * blockDim.y;
	iy = irow + blockIdx.x * blockDim.x;
	unsigned int transform_out_idx = ix + iy * ny;

	if (ix < nx && iy < ny) {
		tile[threadIdx.y][threadIdx.x] = in[transform_in_idx];
		__syncthreads();
		out[transform_out_idx] = tile[icol][irow];
	}
}

__global__ void transform_smem_pad(float* in, float* out, int nx, int ny)
{
	__shared__ float tile[BDIMY][BDIMX+IPAD];
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int transform_in_idx = ix + iy * nx;

	unsigned int bidx = threadIdx.x + threadIdx.y * blockDim.x;
	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.y;

	ix = icol + blockIdx.y * blockDim.y;
	iy = irow + blockIdx.x * blockDim.x;
	unsigned int transform_out_idx = ix + iy * ny;

	if (ix < nx && iy < ny) {
		tile[threadIdx.y][threadIdx.x] = in[transform_in_idx];
		__syncthreads();
		out[transform_out_idx] = tile[icol][irow];
	}
}

__global__ void transform_smem_unroll_pad(float* in, float* out, int nx, int ny)
{
	__shared__ float tile[BDIMY * (BDIMX * 2 + IPAD)];

	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x * 2;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int transform_in_idx = ix + iy * nx;

	unsigned int bidx = threadIdx.x + threadIdx.y * blockDim.x;
	unsigned int irow = bidx / blockDim.y;
	unsigned int icol = bidx % blockDim.y;

	unsigned int ix2 = icol + blockIdx.y * blockDim.y;
	unsigned int iy2 = irow + blockIdx.x * blockDim.x * 2;
	unsigned int transform_out_idx = ix2 + iy2 * ny;

	if (ix + blockDim.x < nx && iy < ny) {
		unsigned int row_idx = threadIdx.x + threadIdx.y * (blockDim.x * 2 + IPAD);
		tile[row_idx] = in[transform_in_idx];
		tile[row_idx + BDIMX] = in[transform_in_idx + BDIMX];
		__syncthreads();
		unsigned int col_idx = irow + icol * (blockDim.x * 2 + IPAD);
		out[transform_out_idx] = tile[col_idx];
		out[transform_out_idx + ny * BDIMX] = tile[col_idx + BDIMX];
	}
}

int main(int argc, char** argv)
{
	printf("strating...\n");

	init_device(0);
	int nx = 1 << 12;
	int ny = 1 << 12;
	int dimx = BDIMX;
	int dimy = BDIMY;
	int nxy = nx * ny;
	int n_bytes = nxy * sizeof(float);
	int transform_kernel = 0;

	if (argc == 2) {
		transform_kernel = atoi(argv[1]);
	}

	if (argc >= 4) {
		transform_kernel = atoi(argv[1]);
		dimx = atoi(argv[2]);
		dimy = atoi(argv[3]);
	}

	float* a_host = (float*)malloc(n_bytes);
	float* b_host_cpu = (float*)malloc(n_bytes);
	float* b_host = (float*)malloc(n_bytes);
	initial_data(a_host, nxy);

	float* a_dev = NULL;
	float* b_dev = NULL;
	CHECK(cudaMalloc((void**)&a_dev, n_bytes));
	CHECK(cudaMalloc((void**)&b_dev, n_bytes));

	CHECK(cudaMemcpy(a_dev, a_host, n_bytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemset(b_dev, 0, n_bytes));

	double start = cpu_second();
	transform_matrix2d_cpu(a_host, b_host_cpu, nx, ny);
	double elaps = cpu_second() - start;
	printf("CPU Execution Time elapsed %f sec\n", elaps);

	dim3 block(dimx, dimy);
	dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);
	dim3 block_1(dimx, dimy);
	dim3 grid_1((nx - 1) / (block_1.x * 2) + 1, (ny - 1) / block_1.y + 1);

	warmup << <grid, block >> > (a_dev, b_dev, nx, ny);
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	switch (transform_kernel)
	{
	case 0:
		copy_row << <grid, block >> > (a_dev, b_dev, nx, ny);
		printf("copy_row ");
		break;
	case 1:
		transform_naive_row << <grid, block >> > (a_dev, b_dev, nx, ny);
		printf("transform_naive_row ");
		break;
	case 2:
		transform_smem << <grid, block >> > (a_dev, b_dev, nx, ny);
		printf("transform_smem ");
		break;
	case 3:
		transform_smem_pad << <grid, block >> > (a_dev, b_dev, nx, ny);
		printf("transform_smem_pad ");
		break;
	case 4:
		transform_smem_unroll_pad << <grid_1, block_1 >> > (a_dev, b_dev, nx, ny);
		printf("transform_smem_unroll_pad ");
		break;
	default:
		break;
	}
	CHECK(cudaDeviceSynchronize());
	elaps = cpu_second() - start;
	printf(" Time elapsed %f sec\n", elaps);
	CHECK(cudaMemcpy(b_host, b_dev, n_bytes, cudaMemcpyDeviceToHost));
	check_result(b_host, b_host_cpu, nxy);

	cudaFree(a_dev);
	cudaFree(b_dev);
	free(a_host);
	free(b_host);
	free(b_host_cpu);
	cudaDeviceReset();

	return 0;
}