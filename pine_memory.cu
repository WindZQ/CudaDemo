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

int main(int argc, char **argv)
{
	int dev = 0;
	cudaSetDevice(dev);

	int elem = 1 << 14;
	printf("Vector size:%d\n", elem);
	int n_byte = sizeof(float) * elem;
	float* a_h = (float*)malloc(n_byte);
	float* b_h = (float*)malloc(n_byte);
	float* res_h = (float*)malloc(n_byte);
	float* res_from_gpu_h = (float*)malloc(n_byte);
	memset(res_h, 0, n_byte);
	memset(res_from_gpu_h, 0, n_byte);

	float* a_d, * b_d, * res_d;

	CHECK(cudaMallocHost((float**)&a_d, n_byte));
	CHECK(cudaMallocHost((float**)&b_d, n_byte));
	CHECK(cudaMallocHost((float**)&res_d, n_byte));

	initial_data(a_h, elem);
	initial_data(b_h, elem);

	CHECK(cudaMemcpy(a_d, a_h, n_byte, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(b_d, b_h, n_byte, cudaMemcpyHostToDevice));

	dim3 block(1024);
	dim3 grid(elem / block.x);
	sum_arrays_gpu << <grid, block >> > (a_d, b_d, res_d);
	printf("Execution configuration<<<%d,%d>>>\n", grid.x, block.x);

	CHECK(cudaMemcpy(res_from_gpu_h, res_d, n_byte, cudaMemcpyDeviceToHost));
	sum_arrays(a_h, b_h, res_h, elem);

	check_result(res_h, res_from_gpu_h, elem);
	cudaFreeHost(a_d);
	cudaFreeHost(b_d);
	cudaFreeHost(res_d);

	free(a_h);
	free(b_h);
	free(res_h);
	free(res_from_gpu_h);

	return 0;
}