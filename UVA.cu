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

__global__ void sum_arrays_gpu(float* a, float* b, float* res)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	res[i] = a[i] + b[i];
}

int main(int argc, char** argv)
{
	int dev = 0;
	cudaSetDevice(dev);

	int elem = 1 << 14;
	printf("Vector size: %d\n", elem);
	int n_byte = sizeof(float) * elem;
	float* res_from_gpu_h = (float*)malloc(n_byte);
	float* res_h = (float*)malloc(n_byte);
	memset(res_h, 0, n_byte);
	memset(res_from_gpu_h, 0, n_byte);

	float* a_host, * b_host, * res_d;
	CHECK(cudaHostAlloc((float**)&a_host, n_byte, cudaHostAllocMapped));
	CHECK(cudaHostAlloc((float**)&b_host, n_byte, cudaHostAllocMapped));
	CHECK(cudaMalloc((float**)&res_d, n_byte));
	res_from_gpu_h = (float*)malloc(n_byte);

	initial_data(a_host, elem);
	initial_data(b_host, elem);

	dim3 block(1024);
	dim3 grid(elem / block.x);
	sum_arrays_gpu << <grid, block >> > (a_host, b_host, res_d);
	printf("Execution configuration<<<%d, %d>>>\n", grid.x, block.x);

	CHECK(cudaMemcpy(res_from_gpu_h, res_d, n_byte, cudaMemcpyDeviceToHost));
	sum_arrays(a_host, b_host, res_h, elem);

	check_result(res_h, res_from_gpu_h, elem);
	cudaFreeHost(a_host);
	cudaFreeHost(b_host);
	cudaFree(res_d);

	free(res_h);
	free(res_from_gpu_h);

	return 0;
}