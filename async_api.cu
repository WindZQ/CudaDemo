#include "freshman.h"

#define N_REPEAT 10
#define N_SEGMENT 1

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

__global__ void sum_array_gpu(float* a, float* b, float* res, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		for (int i = 0; i < N_REPEAT; ++i)
		{
			res[idx] = a[idx] + b[idx];
		}
	}
}

int main(int argc, char** argv)
{
	init_device(0);
	double i_start, elaps;
	i_start = cpu_second();
	int n_elem = 1 << 24;
	printf("Vector size: %d\n", n_elem);
	int n_bytes = sizeof(float) * n_elem;
	float* a_h, * b_h, * res_h, * res_from_gpu_h;
	CHECK(cudaHostAlloc((float**)&a_h, n_bytes, cudaHostAllocDefault));
	CHECK(cudaHostAlloc((float**)&b_h, n_bytes, cudaHostAllocDefault));
	CHECK(cudaHostAlloc((float**)&res_h, n_bytes, cudaHostAllocDefault));
	CHECK(cudaHostAlloc((float**)&res_from_gpu_h, n_bytes, cudaHostAllocDefault));

	cudaMemset(res_h, 0, n_bytes);
	cudaMemset(res_from_gpu_h, 0, n_bytes);

	float* a_d, * b_d, * res_d;
	CHECK(cudaMalloc((float**)&a_d, n_bytes));
	CHECK(cudaMalloc((float**)&b_d, n_bytes));
	CHECK(cudaMalloc((float**)&res_d, n_bytes));

	initial_data(a_h, n_elem);
	initial_data(b_h, n_elem);

	sum_arrays(a_h, b_h, res_h, n_elem);
	dim3 block(512);
	dim3 grid((n_elem - 1) / block.x + 1);

	int i_elem = n_elem / N_SEGMENT;
	cudaStream_t stream[N_SEGMENT];
	for (int i = 0; i < N_SEGMENT; ++i)
	{
		CHECK(cudaStreamCreate(&stream[i]));
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	for (int i = 0; i < N_SEGMENT; ++i)
	{
		int offset = i * i_elem;
		CHECK(cudaMemcpyAsync(&a_d[offset], &a_h[offset], n_bytes / N_SEGMENT, cudaMemcpyHostToDevice, stream[i]));
		CHECK(cudaMemcpyAsync(&b_d[offset], &b_h[offset], n_bytes / N_SEGMENT, cudaMemcpyHostToDevice, stream[i]));
		sum_array_gpu << <grid, block, 0, stream[i] >> > (&a_d[offset], &b_d[offset], &res_d[offset], i_elem);
		CHECK(cudaMemcpyAsync(&res_from_gpu_h[offset], &res_d[offset], n_bytes / N_SEGMENT, cudaMemcpyDeviceToHost, stream[i]));
	}

	CHECK(cudaEventRecord(stop, 0));
	int counter = 0;
	while (cudaEventQuery(stop) == cudaErrorNotReady)
	{
		counter++;
	}
	printf("cpu counter: %d\n", counter);
	elaps = cpu_second() - i_start;
	printf("Asynchronous Execution configuration <<<%d, %d>>> Time elapsed %f sec\n", grid.x, block.x, elaps);
	check_result(res_h, res_from_gpu_h, n_elem);

	for (int i = 0; i < N_SEGMENT; i++)
	{
		CHECK(cudaStreamDestroy(stream[i]));
	}
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(a_h);
	cudaFree(b_h);
	cudaFree(res_h);
	cudaFree(res_from_gpu_h);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}