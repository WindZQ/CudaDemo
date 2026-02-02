#include "freshman.h"

void sum_matrix2d_cpu(float* mat_a, float* mat_b, float* mat_c, int nx, int ny)
{
	for (int j = 0; j < ny; ++j)
	{
		for (int i = 0; i < nx; ++i)
		{
			mat_c[i] = mat_a[i] + mat_b[i];
		}

		mat_c += nx;
		mat_b += nx;
		mat_a += nx;
	}
}

__global__ void sum_matrix(float* mat_a, float* mat_b, float* mat_c, int nx, int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int idx = ix + iy * ny;

	if (ix < nx && iy < ny) {
		mat_c[idx] = mat_a[idx] + mat_b[idx];
	}
}

int main(int argc, char** argv)
{
	int nx = 1 << 13;
	int ny = 1 << 13;
	int nxy = nx * ny;
	int n_bytes = nxy * sizeof(float);

	float* a_host = (float*)malloc(n_bytes);
	float* b_host = (float*)malloc(n_bytes);
	float* c_host = (float*)malloc(n_bytes);
	float* c_from_gpu = (float*)malloc(n_bytes);
	initial_data(a_host, nxy);
	initial_data(b_host, nxy);

	float* a_dev = NULL;
	float* b_dev = NULL;
	float* c_dev = NULL;
	CHECK(cudaMalloc((void**)&a_dev, n_bytes));
	CHECK(cudaMalloc((void**)&b_dev, n_bytes));
	CHECK(cudaMalloc((void**)&c_dev, n_bytes));


	CHECK(cudaMemcpy(a_dev, a_host, n_bytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(b_dev, b_host, n_bytes, cudaMemcpyHostToDevice));

	int dimx = argc > 2 ? atoi(argv[1]) : 32;
	int dimy = argc > 2 ? atoi(argv[2]) : 32;

	double start, elaps;
	start = cpu_second();
	sum_matrix2d_cpu(a_host, b_host, c_host, nx, ny);
	elaps = cpu_second() - start;
	printf("CPU Execution Time elapsed %f sec\n", elaps);

	dim3 block_0(32, 32);
	dim3 grid_0((nx - 1) / block_0.x + 1, (ny - 1) / block_0.y + 1);
	start = cpu_second();
	sum_matrix << <grid_0, block_0 >> > (a_dev, b_dev, c_dev, nx, ny);
	CHECK(cudaDeviceSynchronize());
	printf("Warm Up \n");

	dim3 block(dimx, dimy);
	dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1);
	start = cpu_second();
	sum_matrix << <grid, block >> > (a_dev, b_dev, c_dev, nx, ny);
	CHECK(cudaDeviceSynchronize());
	elaps = cpu_second() - start;
	printf("GPU Execution configuration<<<(%d,%d),(%d,%d)>>> Time elapsed %f sec\n",
		grid.x, grid.y, block.x, block.y, elaps);
	CHECK(cudaMemcpy(c_from_gpu, c_dev, n_bytes, cudaMemcpyDeviceToHost));

	check_result(c_host, c_from_gpu, nxy);

	cudaFree(a_dev);
	cudaFree(b_dev);
	cudaFree(c_dev);
	free(a_host);
	free(b_host);
	free(c_host);
	free(c_from_gpu);
	cudaDeviceReset();

	return 0;
}