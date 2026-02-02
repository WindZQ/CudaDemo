#include <cooperative_groups.h>

#include "freshman.h"

int recursive_reduce(int* data, int const size)
{
	if (size == 1) return data[0];

	int const stride = size / 2;
	if (size % 2 == 1) {
		for (int i = 0; i < stride; ++i)
		{
			data[i] = data[i + stride];
		}

		data[0] += data[size - 1];
	} else {
		for (int i = 0; i < stride; ++i)
		{
			data[i] += data[i + stride];
		}
	}

	return recursive_reduce(data, stride);
}

__global__ void warmup(int* g_idata, int *g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	if (tid >= n) return;

	int* idata = g_idata + blockIdx.x * blockDim.x;

	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride) == 0)) {
			idata[tid] += idata[tid + stride];
		}

		__syncthreads();
	}

	if (tid == 0) {
		g_odata[blockIdx.x] = idata[0];
	}
}

__global__ void reduce_neighbored(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;

	if (tid >= n) return;

	int* idata = g_idata + blockIdx.x * blockDim.x;
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride) == 0)) {
			idata[tid] += idata[tid + stride];
		}

		__syncthreads();
	}

	if (tid == 0) {
		g_odata[blockIdx.x] = idata[0];
	}
}

__global__ void reduce_neighbored_less(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

	int* idata = g_idata + blockIdx.x * blockDim.x;
	if (idx > n) return;
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		int index = 2 * stride * tid;
		if (index < blockDim.x) {
			idata[index] += idata[index + stride];
		}
		__syncthreads();
	}

	if (tid == 0) {
		g_odata[blockIdx.x] = idata[0];
	}
}

__global__ void reduce_interleaved(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

	int* idata = g_idata + blockIdx.x * blockDim.x;
	if (idx >= n) return;
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{

		if (tid < stride) {
			idata[tid] += idata[tid + stride];
		}

		__syncthreads();
	}

	if (tid == 0) {
		g_odata[blockIdx.x] = idata[0];
	}

}

int main(int argc, char** argv)
{
	init_device(0);

	bool result = false;

	int size = 1 << 24;
	printf("	with array size %d  ", size);

	int blocksize = 1024;
	if (argc > 1)
	{
		blocksize = atoi(argv[1]);
	}

	dim3 block(blocksize, 1);
	dim3 grid((size - 1) / block.x + 1, 1);
	printf("grid %d block %d \n", grid.x, block.x);

	size_t bytes = size * sizeof(int);
	int* idata_host = (int*)malloc(bytes);
	int* odata_host = (int*)malloc(grid.x * sizeof(int));
	int* tmp = (int*)malloc(bytes);

	initial_data_int(idata_host, size);

	memcpy(tmp, idata_host, bytes);
	double start, elaps;
	int gpu_sum = 0;

	int* idata_dev = NULL;
	int* odata_dev = NULL;
	CHECK(cudaMalloc((void**)&idata_dev, bytes));
	CHECK(cudaMalloc((void**)&odata_dev, grid.x * sizeof(int)));

	int cpu_sum = 0;
	start = cpu_second();
	//cpu_sum = recursive_reduce(tmp, size);
	for (int i = 0; i < size; i++)
		cpu_sum += tmp[i];
	printf("cpu sum:%d \n", cpu_sum);
	elaps = cpu_second() - start;
	printf("cpu reduce                 elapsed %lf ms cpu_sum: %d\n", elaps, cpu_sum);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	warmup << <grid, block >> > (idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu warmup                 elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
		elaps, gpu_sum, grid.x, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	reduce_neighbored << <grid, block >> > (idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduce_neighbored       elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
		elaps, gpu_sum, grid.x, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	reduce_neighbored_less << <grid, block >> > (idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduce_neighbored_less   elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
		elaps, gpu_sum, grid.x, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	reduce_interleaved << <grid, block >> > (idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduce_interleaved      elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
		elaps, gpu_sum, grid.x, block.x);

	free(idata_host);
	free(odata_host);
	CHECK(cudaFree(idata_dev));
	CHECK(cudaFree(odata_dev));

	cudaDeviceReset();

	if (gpu_sum == cpu_sum) {
		printf("Test success!\n");
	}

	return 0;
}