#include "freshman.h"

int recursive_reduce(int* data, int const size)
{
	if (size == 1) return data[0];

	int const stride = size / 2;
	if (size % 2 == 1) {
		for (int i = 0; i < stride; ++i)
		{
			data[i] += data[i + stride];
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

__global__ void warmup(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	if (tid >= n) return;

	int* idata = g_idata + blockIdx.x * blockDim.x;
	for (int stride = 1; stride < blockDim.x; ++stride)
	{
		if ((tid % (2 * stride)) == 0) {
			idata[tid] += idata[tid + stride];
		}

		__syncthreads();
	}

	if (tid == 0) {
		g_odata[blockIdx.x] = idata[0];
	}
}

__global__ void reduce_unroll2(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;

	if (tid >= n) return;
	int* idata = g_idata + blockIdx.x * blockDim.x * 2;
	if (idx + blockDim.x < n) {
		g_idata[idx] += g_idata[idx + blockDim.x];
	}

	__syncthreads();

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

__global__ void reduce_unroll4(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x * 4 + threadIdx.x;

	if (tid >= n) return;
	int* idata = g_idata + blockIdx.x * blockDim.x * 4;
	if (idx + blockDim.x < n) {
		g_idata[idx] += g_idata[idx + blockDim.x];
		g_idata[idx] += g_idata[idx + blockDim.x * 2];
		g_idata[idx] += g_idata[idx + blockDim.x * 3];
	}

	__syncthreads();

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

__global__ void reduce_unroll8(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;

	if (tid >= n) return;
	int* idata = g_idata + blockIdx.x * blockDim.x * 8;
	if (idx + blockDim.x < n) {
		g_idata[idx] += g_idata[idx + blockDim.x];
		g_idata[idx] += g_idata[idx + blockDim.x * 2];
		g_idata[idx] += g_idata[idx + blockDim.x * 3];
		g_idata[idx] += g_idata[idx + blockDim.x * 4];
		g_idata[idx] += g_idata[idx + blockDim.x * 5];
		g_idata[idx] += g_idata[idx + blockDim.x * 6];
		g_idata[idx] += g_idata[idx + blockDim.x * 7];
		g_idata[idx] += g_idata[idx + blockDim.x * 8];
	}

	__syncthreads();

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

__global__ void reduce_unroll_warp8(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;

	if (tid >= n) return;

	int* idata = g_idata + blockIdx.x * blockDim.x * 8;
	if (idx + 7 * blockDim.x < n) {
		int a1 = g_idata[idx];
		int a2 = g_idata[idx + blockDim.x];
		int a3 = g_idata[idx + 2 * blockDim.x];
		int a4 = g_idata[idx + 3 * blockDim.x];
		int a5 = g_idata[idx + 4 * blockDim.x];
		int a6 = g_idata[idx + 5 * blockDim.x];
		int a7 = g_idata[idx + 6 * blockDim.x];
		int a8 = g_idata[idx + 7 * blockDim.x];
		g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
	}

	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
	{
		if (tid < stride) {
			idata[tid] += idata[tid + stride];
		}

		__syncthreads();
	}

	if (tid < 32) {
		volatile int* vmem = idata;
		vmem[tid] += vmem[tid + 32];
		vmem[tid] += vmem[tid + 16];
		vmem[tid] += vmem[tid + 8];
		vmem[tid] += vmem[tid + 4];
		vmem[tid] += vmem[tid + 2];
		vmem[tid] += vmem[tid + 1];
	}

	if (tid == 0) {
		g_odata[blockIdx.x] = idata[0];
	}
}

__global__ void reduce_complete_unroll_warp8(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;

	if (tid >= n) return;

	int* idata = g_idata + blockIdx.x * blockDim.x * 8;
	if (idx + 7 * blockDim.x < n) {
		int a1 = g_idata[idx];
		int a2 = g_idata[idx + blockDim.x];
		int a3 = g_idata[idx + 2 * blockDim.x];
		int a4 = g_idata[idx + 3 * blockDim.x];
		int a5 = g_idata[idx + 4 * blockDim.x];
		int a6 = g_idata[idx + 5 * blockDim.x];
		int a7 = g_idata[idx + 6 * blockDim.x];
		int a8 = g_idata[idx + 7 * blockDim.x];
		g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
	}
	__syncthreads();

	if (blockDim.x >= 1024 && tid < 512) {
		idata[tid] += idata[tid + 512];
	}
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256) {
		idata[tid] += idata[tid + 256];
	}
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128) {
		idata[tid] += idata[tid + 128];
	}
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64) {
		idata[tid] += idata[tid + 64];
	}
	__syncthreads();

	if (tid < 32) {
		volatile int* vmem = idata;
		vmem[tid] += vmem[tid + 32];
		vmem[tid] += vmem[tid + 16];
		vmem[tid] += vmem[tid + 8];
		vmem[tid] += vmem[tid + 4];
		vmem[tid] += vmem[tid + 2];
		vmem[tid] += vmem[tid + 1];
	}

	if (tid == 0) {
		g_odata[blockIdx.x] = idata[0];
	}
}

template <unsigned int iBlockSize>
__global__ void reduce_complete_unroll(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;

	if (tid >= n) return;

	int* idata = g_idata + blockIdx.x * blockDim.x * 8;
	if (idx + 7 * blockDim.x < n) {
		int a1 = g_idata[idx];
		int a2 = g_idata[idx + blockDim.x];
		int a3 = g_idata[idx + 2 * blockDim.x];
		int a4 = g_idata[idx + 3 * blockDim.x];
		int a5 = g_idata[idx + 4 * blockDim.x];
		int a6 = g_idata[idx + 5 * blockDim.x];
		int a7 = g_idata[idx + 6 * blockDim.x];
		int a8 = g_idata[idx + 7 * blockDim.x];
		g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
	}
	__syncthreads();

	if (iBlockSize >= 1024 && tid < 512) {
		idata[tid] += idata[tid + 512];
	}
	__syncthreads();

	if (iBlockSize >= 512 && tid < 256) {
		idata[tid] += idata[tid + 256];
	}
	__syncthreads();

	if (iBlockSize >= 256 && tid < 128) {
		idata[tid] += idata[tid + 128];
	}
	__syncthreads();

	if (iBlockSize >= 128 && tid < 64) {
		idata[tid] += idata[tid + 64];
	}
	__syncthreads();

	if (tid < 32) {
		volatile int* vmem = idata;
		vmem[tid] += vmem[tid + 32];
		vmem[tid] += vmem[tid + 16];
		vmem[tid] += vmem[tid + 8];
		vmem[tid] += vmem[tid + 4];
		vmem[tid] += vmem[tid + 2];
		vmem[tid] += vmem[tid + 1];
	}

	if (tid == 0) {
		g_odata[blockIdx.x] = idata[0];
	}	
}

int main(int argc, char** argv)
{
	init_device(0);
	int size = 1 << 24;
	printf("	with array size %d  ", size);

	int blocksize = 1024;
	if (argc > 1) {
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

	for (int i = 0; i < size; i++)
		cpu_sum += tmp[i];
	printf("cpu sum:%d \n", cpu_sum);
	elaps = cpu_second() - start;
	printf("cpu reduce                  elapsed %lf ms cpu_sum: %d\n", elaps, cpu_sum);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	warmup << <grid.x / 2, block >> > (idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	printf("gpu warmup                  elapsed %lf ms \n", elaps);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	reduce_unroll2 << <grid.x / 2, block >> > (idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x / 2; i++)
		gpu_sum += odata_host[i];
	printf("reduce_unrolling2            elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		elaps, gpu_sum, grid.x / 2, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	reduce_unroll4 << <grid.x / 4, block >> > (idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x / 4; i++)
		gpu_sum += odata_host[i];
	printf("reduce_unrolling4            elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		elaps, gpu_sum, grid.x / 4, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	reduce_unroll8 << <grid.x / 8, block >> > (idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x / 8; i++)
		gpu_sum += odata_host[i];
	printf("reduce_unrolling8            elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		elaps, gpu_sum, grid.x / 8, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	reduce_unroll_warp8 << <grid.x / 8, block >> > (idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x / 8; i++)
		gpu_sum += odata_host[i];
	printf("reduce_unrolling_warp8        elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		elaps, gpu_sum, grid.x / 8, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	reduce_complete_unroll_warp8 << <grid.x / 8, block >> > (idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x / 8; i++)
		gpu_sum += odata_host[i];
	printf("reduce_complete_unroll_warp8   elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		elaps, gpu_sum, grid.x / 8, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	switch (blocksize)
	{
	case 1024:
		reduce_complete_unroll <1024> << <grid.x / 8, block >> > (idata_dev, odata_dev, size);
		break;
	case 512:
		reduce_complete_unroll <512> << <grid.x / 8, block >> > (idata_dev, odata_dev, size);
		break;
	case 256:
		reduce_complete_unroll <256> << <grid.x / 8, block >> > (idata_dev, odata_dev, size);
		break;
	case 128:
		reduce_complete_unroll <128> << <grid.x / 8, block >> > (idata_dev, odata_dev, size);
		break;
	}
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x / 8; i++)
		gpu_sum += odata_host[i];
	printf("reduce_complete_unroll        elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
		elaps, gpu_sum, grid.x / 8, block.x);

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