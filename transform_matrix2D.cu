#include "freshman.h"

void transform_matrix2d_cpu(float* mat_a, float* mat_b, int nx, int ny)
{
	for (int j = 0; j < ny; ++j)
	{
		for (int i = 0; i < nx; ++i)
		{
			mat_b[i * nx + j] = mat_a[j * nx + i];
		}
	}
}

__global__ void copy_row(float* mat_a, float* mat_b, int nx, int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = ix + iy * nx;

	if (ix < nx && iy < ny) {
		mat_b[idx] = mat_a[idx];
	}
}

__global__ void copy_col(float* mat_a, float* mat_b, int nx, int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = ix * ny + iy;

	if (ix < nx && iy < ny) {
		mat_b[idx] = mat_a[idx];
	}
}

__global__ void transform_naive_row(float* mat_a, float* mat_b, int nx, int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int idx_row = ix + iy * nx;
	int idx_col = ix * ny + iy;

	if (ix < nx && iy < ny) {
		mat_b[idx_col] = mat_a[idx_row];
	}
}

__global__ void transform_naive_col(float* mat_a, float* mat_b, int nx, int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int idx_row = ix + iy * nx;
	int idx_col = ix * ny + iy;

	if (ix < nx && iy < ny) {
		mat_b[idx_row] = mat_a[idx_col];
	}
}

__global__ void transform_naive_row_unroll(float* mat_a, float* mat_b, int nx, int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x * 4;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int idx_row = ix + iy * nx;
	int idx_col = ix * ny + iy;

	if (ix < nx && iy < ny) {
		mat_b[idx_col] = mat_a[idx_row];
		mat_b[idx_col + ny * 1 * blockDim.x] = mat_a[idx_row + 1 * blockDim.x];
		mat_b[idx_col + ny * 2 * blockDim.x] = mat_a[idx_row + 2 * blockDim.x];
		mat_b[idx_col + ny * 3 * blockDim.x] = mat_a[idx_row + 3 * blockDim.x];
	}
}

__global__ void transform_naive_col_unroll(float* mat_a, float* mat_b, int nx, int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x * 4;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int idx_row = ix + iy * nx;
	int idx_col = ix * ny + iy;

	if (ix < nx && iy < ny) {
		mat_b[idx_row] = mat_a[idx_col];
		mat_b[idx_row + 1 * blockDim.x] = mat_a[idx_col + ny * 1 * blockDim.x];
		mat_b[idx_row + 2 * blockDim.x] = mat_a[idx_col + ny * 2 * blockDim.x];
		mat_b[idx_row + 3 * blockDim.x] = mat_a[idx_col + ny * 3 * blockDim.x];
	}
}

__global__ void transform_naive_row_diagonal(float* mat_a, float* mat_b, int nx, int ny)
{
	int block_x = blockIdx.x;
	int block_y = (blockIdx.x + blockIdx.y) % gridDim.x;
	int ix = threadIdx.x + blockDim.x * block_x;
	int iy = threadIdx.y + blockDim.y * block_y;
	int idx_row = ix + iy * nx;
	int idx_col = ix * ny + iy;

	if (ix < nx && iy < ny) {
		mat_b[idx_col] = mat_a[idx_row];
	}
}

__global__ void transform_naive_col_diagonal(float* mat_a, float* mat_b, int nx, int ny)
{
	int block_x = blockIdx.x;
	int block_y = (blockIdx.x + blockIdx.y) % gridDim.x;
	int ix = threadIdx.x + blockDim.x * block_x;
	int iy = threadIdx.y + blockDim.y * block_y;
	int idx_row = ix + iy * nx;
	int idx_col = ix * ny + iy;

	if (ix < nx && iy < ny) {
		mat_b[idx_row] = mat_a[idx_col];
	}
}

int main(int argc, char** argv)
{
	printf("strating...\n");
	init_device(0);
	int nx = 1 << 12;
	int ny = 1 << 12;
	int dimx = 32;
	int dimy = 32;
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
	float* b_host = (float*)malloc(n_bytes);
	initial_data(a_host, nxy);

	float* a_dev = NULL;
	float* b_dev = NULL;
	CHECK(cudaMalloc((void**)&a_dev, n_bytes));
	CHECK(cudaMalloc((void**)&b_dev, n_bytes));

	CHECK(cudaMemcpy(a_dev, a_host, n_bytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemset(b_dev, 0, n_bytes));

	double start = cpu_second();
	transform_matrix2d_cpu(a_host, b_host, nx, ny);
	double elaps = cpu_second() - start;
	printf("CPU Execution Time elapsed %f sec\n", elaps);

	dim3 block(dimx, dimy);
	dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);
	dim3 block_1(dimx, dimy);
	dim3 grid_1((nx - 1) / (block_1.x * 4) + 1, (ny - 1) / block_1.y + 1);
	start = cpu_second();
	switch (transform_kernel)
	{
	case 0:
		copy_row << <grid, block >> > (a_dev, b_dev, nx, ny);
		break;
	case 1:
		copy_col << <grid, block >> > (a_dev, b_dev, nx, ny);
		break;
	case 2:
		transform_naive_row << <grid, block >> > (a_dev, b_dev, nx, ny);
		break;
	case 3:
		transform_naive_col << <grid, block >> > (a_dev, b_dev, nx, ny);
		break;
	case 4:
		transform_naive_col_unroll << <grid_1, block_1 >> > (a_dev, b_dev, nx, ny);
		break;
	case 5:
		transform_naive_col_unroll << <grid_1, block_1 >> > (a_dev, b_dev, nx, ny);
		break;
	case 6:
		transform_naive_row_diagonal << <grid, block >> > (a_dev, b_dev, nx, ny);
		break;
	case 7:
		transform_naive_col_diagonal << <grid, block >> > (a_dev, b_dev, nx, ny);
		break;
	default:
		break;
	}
	CHECK(cudaDeviceSynchronize());
	elaps = cpu_second() - start;
	printf(" Time elapsed %f sec\n", elaps);
	CHECK(cudaMemcpy(b_host, b_dev, n_bytes, cudaMemcpyDeviceToHost));
	check_result(b_host, b_host, nxy);

	cudaFree(a_dev);
	cudaFree(b_dev);
	free(a_host);
	free(b_host);
	cudaDeviceReset();

	return 0;
}