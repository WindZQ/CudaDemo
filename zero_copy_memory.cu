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
	int power = 10;
	if (argc >= 2)
		power = atoi(argv[1]);

	int elem = 1 << power;
	printf("Vector size:%d\n", elem);
	int n_byte = sizeof(float) * elem;
	float* res_from_gpu_h = (float*)malloc(n_byte);
	float* res_h = (float*)malloc(n_byte);
	memset(res_h, 0, n_byte);
	memset(res_from_gpu_h, 0, n_byte);

	float* a_host, * b_host, * res_d;
	double start, elaps;
	dim3 block(1024);
	dim3 grid(elem / block.x);
	res_from_gpu_h = (float*)malloc(n_byte);
	float* a_dev, * b_dev;
	CHECK(cudaHostAlloc((float**)&a_host, n_byte, cudaHostAllocMapped));
	CHECK(cudaHostAlloc((float**)&b_host, n_byte, cudaHostAllocMapped));
	CHECK(cudaMalloc((float**)&res_d, n_byte));
	initial_data(a_host, elem);
	initial_data(b_host, elem);

	start = cpu_second();
	CHECK(cudaHostGetDevicePointer((void**)&a_dev, (void*)a_host, 0));
	CHECK(cudaHostGetDevicePointer((void**)&b_dev, (void*)b_host, 0));
	sum_arrays_gpu << <grid, block >> > (a_dev, b_dev, res_d);
	CHECK(cudaMemcpy(res_from_gpu_h, res_d, n_byte, cudaMemcpyDeviceToHost));
	elaps = cpu_second() - start;
	printf("zero copy memory elapsed %lf ms \n", elaps);
	printf("Execution configuration<<<%d,%d>>>\n", grid.x, block.x);
	float* a_h_n = (float*)malloc(n_byte);
	float* b_h_n = (float*)malloc(n_byte);
	float* res_h_n = (float*)malloc(n_byte);
	float* res_from_gpu_h_n = (float*)malloc(n_byte);
	memset(res_h_n, 0, n_byte);
	memset(res_from_gpu_h_n, 0, n_byte);

	float* a_d_n, * b_d_n, * res_d_n;
	CHECK(cudaMalloc((float**)&a_d_n, n_byte));
	CHECK(cudaMalloc((float**)&b_d_n, n_byte));
	CHECK(cudaMalloc((float**)&res_d_n, n_byte));

	initial_data(a_h_n, elem);
	initial_data(b_h_n, elem);
	start = cpu_second();
	CHECK(cudaMemcpy(a_d_n, a_h_n, n_byte, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(b_d_n, b_h_n, n_byte, cudaMemcpyHostToDevice));
	sum_arrays_gpu << <grid, block >> > (a_d_n, b_d_n, res_d_n);
	CHECK(cudaMemcpy(res_from_gpu_h, res_d, n_byte, cudaMemcpyDeviceToHost));
	elaps = cpu_second() - start;
	printf("device memory elapsed %lf ms \n", elaps);
	printf("Execution configuration<<<%d,%d>>>\n", grid.x, block.x);

	sum_arrays(a_host, b_host, res_h, elem);
	check_result(res_h, res_from_gpu_h, elem);

	cudaFreeHost(a_host);
	cudaFreeHost(b_host);
	cudaFree(res_d);
	free(res_h);
	free(res_from_gpu_h);

	cudaFree(a_d_n);
	cudaFree(b_d_n);
	cudaFree(res_d_n);

	free(a_h_n);
	free(b_h_n);
	free(res_h_n);
	free(res_from_gpu_h_n);

	return 0;
}