#include "freshman.h"

__global__ void nesthelloworld(int size, int depth)
{
	unsigned int tid = threadIdx.x;
	printf("depth : %d blockIdx: %d,threadIdx: %d\n", depth, blockIdx.x, threadIdx.x);

	if (size == 1) return;

	int nthread = (size >> 1);
	if (tid == 0 && nthread > 0) {
		nesthelloworld << <1, nthread >> > (nthread, ++depth);
		printf("-----------> nested execution depth: %d\n", depth);
	}
}

int main(int argc, char** argv)
{
	int size = 64;
	int block_x = 2;

	dim3 block(block_x, 1);
	dim3 grid((size - 1) / block.x + 1, 1);
	nesthelloworld << <grid, block >> > (size, 0);
	cudaGetLastError();
	cudaDeviceReset();

	return 0;
}