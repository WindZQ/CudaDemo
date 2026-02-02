#include "freshman.h"

void sum_arrays(float* a, float* b, float* res, int offset, const int size)
{
	for (int i = 0, k = offset; k < size; ++i, ++k)
	{
		res[i] = a[k] + b[k];
	}
}

__global__ void sum_arrays_gpu(float* a, float* b, float* res, int offset, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int k = i + offset;

	if (k + 3 * blockDim.x < n) {
		res[i] = a[k] + b[k];
		res[i + blockDim.x] = a[k + blockDim.x] + b[k + blockDim.x];
		res[i + blockDim.x * 2] = a[k + blockDim.x * 2] + b[k + blockDim.x * 2];
		res[i + blockDim.x * 3] = a[k + blockDim.x * 3] + b[k + blockDim.x * 3];
	}
}

int main(int argc, char** argv)
{
	int dev = 0;
	cudaSetDevice(dev);
	int block_x = 512;

	int elem = 1 << 18;
	int offset = 0;
	if (argc == 2) {
		offset = atoi(argv[1]);
	} else if (argc == 3) {
		offset = atoi(argv[1]);
		block_x = atoi(argv[2]);
	}
	printf("Vector size:%d\n", elem);

	int n_byte = sizeof(float) * elem;
	float* a_h = (float*)malloc(n_byte);
	float* b_h = (float*)malloc(n_byte);
	float* res_h = (float*)malloc(n_byte);
	float* res_from_gpu_h = (float*)malloc(n_byte);
	memset(res_h, 0, n_byte);
	memset(res_from_gpu_h, 0, n_byte);

	float* a_d, * b_d, * res_d;
	CHECK(cudaMalloc((float**)&a_d, n_byte));
	CHECK(cudaMalloc((float**)&b_d, n_byte));
	CHECK(cudaMalloc((float**)&res_d, n_byte));
	CHECK(cudaMemset(res_d, 0, n_byte));
	initial_data(a_h, elem);
	initial_data(b_h, elem);

	CHECK(cudaMemcpy(a_d, a_h, n_byte, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(b_d, b_h, n_byte, cudaMemcpyHostToDevice));

	dim3 block(block_x);
	dim3 grid(elem / block.x);
	double start, elaps;
	start = cpu_second();
	sum_arrays_gpu << <grid, block >> > (a_d, b_d, res_d, offset, elem);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;

	printf("warmup Time elapsed %f sec\n", elaps);
	start = cpu_second();
	sum_arrays_gpu << <grid, block >> > (a_d, b_d, res_d, offset, elem);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	CHECK(cudaMemcpy(res_from_gpu_h, res_d, n_byte, cudaMemcpyDeviceToHost));
	printf("Execution configuration <<<%d, %d>>> Time elapsed %f sec --offset:%d \n", grid.x, block.x, elaps, offset);

	sum_arrays(a_h, b_h, res_h, offset, elem);

	check_result(res_h, res_from_gpu_h, elem - 4 * block_x);
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(res_d);

	free(a_h);
	free(b_h);
	free(res_h);
	free(res_from_gpu_h);

	return 0;
}