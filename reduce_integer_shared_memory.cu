#include "freshman.h"

#define DIM 1024 

int recursive_reduce(int* data, int const size)
{
	if (size == 1) return data[0];

	int const stride = size / 2;
	if (stride % 2 == 1) {
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
	for (int stride = 1; stride < blockDim.x; stride *= 2)
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

__global__ void reduce_gmem(int* g_idata, int* g_odata, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= n) return;
	int* idata = g_idata + blockIdx.x * blockDim.x;

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

__global__ void reduce_smem(int* g_idata, int* g_odata, unsigned int n)
{
	__shared__ int smem[DIM];
	unsigned int tid = threadIdx.x;
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= n) return;
	int* idata = g_idata + blockIdx.x * blockDim.x;

	smem[tid] = idata[tid];
	__syncthreads();

	if (blockDim.x >= 1024 && tid < 512) {
		smem[tid] += smem[tid + 512];
	}
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256) {
		smem[tid] += smem[tid + 256];
	}
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128) {
		smem[tid] += smem[tid + 128];
	}
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64) {
		smem[tid] += smem[tid + 64];
	}
	__syncthreads();

	if (tid < 32) {
		volatile int* vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	if (tid == 0) {
		g_odata[blockIdx.x] = smem[0];
	}
}

__global__ void reduce_unroll4_smem(int* g_idata, int* g_odata, unsigned int n)
{
	__shared__ int smem[DIM];
	unsigned int tid = threadIdx.x;
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 4;

	if (tid >= n) return;
	int* idata = g_idata + blockIdx.x * blockDim.x;

	int temp_sum = 0;
	if (idx + 3 * blockDim.x <= n) {
		int a1 = g_idata[idx];
		int a2 = g_idata[idx + blockDim.x];
		int a3 = g_idata[idx + blockDim.x * 2];
		int a4 = g_idata[idx + blockDim.x * 3];
		temp_sum = a1 + a2 + a3 + a4;
	}

	smem[tid] = temp_sum;
	__syncthreads();

	if (blockDim.x >= 1024 && tid < 512) {
		smem[tid] += smem[tid + 512];
	}
	__syncthreads();

	if (blockDim.x >= 512 && tid < 256) {
		smem[tid] += smem[tid + 256];
	}
	__syncthreads();

	if (blockDim.x >= 256 && tid < 128) {
		smem[tid] += smem[tid + 128];
	}
	__syncthreads();

	if (blockDim.x >= 128 && tid < 64) {
		smem[tid] += smem[tid + 64];
	}
	__syncthreads();

	if (tid < 32) {
		volatile int* vsmem = smem;
		vsmem[tid] += vsmem[tid + 32];
		vsmem[tid] += vsmem[tid + 16];
		vsmem[tid] += vsmem[tid + 8];
		vsmem[tid] += vsmem[tid + 4];
		vsmem[tid] += vsmem[tid + 2];
		vsmem[tid] += vsmem[tid + 1];
	}

	if (tid == 0) {
		g_odata[blockIdx.x] = smem[0];
	}
}

int main(int argc, char** argv)
{
	init_device(0);
	int size = 1 << 24;
	printf("	with array size %d  \n", size);

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
	//cpu_sum = recursiveReduce(tmp, size);
	for (int i = 0; i < size; i++)
		cpu_sum += tmp[i];
	elaps = cpu_second() - start;
	printf("cpu reduce           elapsed %lf ms cpu_sum: %d\n", elaps, cpu_sum);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	warmup << <grid.x / 2, block >> > (idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	printf("gpu warmup           elapsed %lf ms\n", elaps);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	reduce_gmem << <grid.x, block >> > (idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("reduce_gmem           elapsed %lf ms gpu_sum: %d\n", elaps, gpu_sum);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	reduce_smem << <grid.x, block >> > (idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("reduce_smem           elapsed %lf ms gpu_sum: %d\n", elaps, gpu_sum);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	start = cpu_second();
	reduce_unroll4_smem << <grid.x / 4, block >> > (idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	elaps = cpu_second() - start;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x / 4; i++)
		gpu_sum += odata_host[i];
	printf("reduce_unroll4_smem    elapsed %lf ms gpu_sum: %d\n", elaps, gpu_sum);

	free(idata_host);
	free(odata_host);
	CHECK(cudaFree(idata_dev));
	CHECK(cudaFree(odata_dev));

	cudaDeviceReset();

	return 0;
}