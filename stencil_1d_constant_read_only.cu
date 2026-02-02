#include "freshman.h"

#define TEMPLATE_SIZE 9
#define TEMP_RADIO_SIZE (TEMPLATE_SIZE / 2)
#define BDIM 32

__constant__ float coef[TEMP_RADIO_SIZE];

void convolution(float* in, float* out, float* in_template, const unsigned int size)
{
	for (int i = TEMP_RADIO_SIZE; i < size - TEMP_RADIO_SIZE; ++i)
	{
		for (int j = 1; j <= TEMP_RADIO_SIZE; ++j)
		{
			out[i] += in_template[j - 1] * (in[i + j] - in[i - j]);
		}
	}
}

__global__ void stencil_1d(float* in, float* out)
{
	__shared__ float smem[BDIM + 2 * TEMP_RADIO_SIZE];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int sidx = threadIdx.x + TEMP_RADIO_SIZE;
	smem[sidx] = in[idx];

	if (threadIdx.x < TEMP_RADIO_SIZE) {
		if (idx > TEMP_RADIO_SIZE) {
			smem[sidx - TEMP_RADIO_SIZE] = in[idx - TEMP_RADIO_SIZE];
		}

		if (idx < gridDim.x * blockDim.x - BDIM) {
			smem[sidx + BDIM] = in[idx + BDIM];
		}
	}

	__syncthreads();
	if (idx < TEMP_RADIO_SIZE || idx >= gridDim.x * blockDim.x - TEMP_RADIO_SIZE) {
		return;
	}

	float temp = 0.0f;
#pragma unroll
	for (int i = 1; i <= TEMP_RADIO_SIZE; ++i)
	{
		temp += coef[i - 1] * (smem[sidx + i] - smem[sidx - i]);
	}

	out[idx] = temp;
}

__global__ void stencil_1d_readonly(float* in, float* out, const float * __restrict__ dcoef)
{
	__shared__ float smem[BDIM + 2 * TEMP_RADIO_SIZE];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int sidx = threadIdx.x + TEMP_RADIO_SIZE;
	smem[sidx] = in[idx];

	if (threadIdx.x < TEMP_RADIO_SIZE) {
		if (idx > TEMP_RADIO_SIZE) {
			smem[sidx - TEMP_RADIO_SIZE] = in[idx - TEMP_RADIO_SIZE];
		}

		if (idx < gridDim.x * blockDim.x - BDIM) {
			smem[sidx + BDIM] = in[idx + BDIM];
		}
	}

	__syncthreads();
	if (idx < TEMP_RADIO_SIZE || idx >= gridDim.x * blockDim.x - TEMP_RADIO_SIZE) {
		return;
	}

	float temp = 0.0f;
#pragma unroll
	for (int i = 1; i <= TEMP_RADIO_SIZE; ++i)
	{
		temp += coef[i - 1] * (smem[sidx + i] - smem[sidx - i]);
	}

	out[idx] = temp;
}

int main(int argc, char** argv)
{
    printf("strating...\n");
    init_device(0);
    int dimx = BDIM;
    unsigned int nxy = 1 << 16;
    int n_bytes = nxy * sizeof(float);

    float* in_host = (float*)malloc(n_bytes);
    float* out_gpu = (float*)malloc(n_bytes);
    float* out_cpu = (float*)malloc(n_bytes);
    memset(out_cpu, 0, n_bytes);
    initial_data(in_host, nxy);

    float* in_dev = NULL;
    float* out_dev = NULL;

    initial_data(in_host, nxy);
    float templ_[] = { -1.0, -2.0, 2.0, 1.0 };
    CHECK(cudaMemcpyToSymbol(coef, templ_, TEMP_RADIO_SIZE * sizeof(float)));

    CHECK(cudaMalloc((void**)&in_dev, n_bytes));
    CHECK(cudaMalloc((void**)&out_dev, n_bytes));
    CHECK(cudaMemcpy(in_dev, in_host, n_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(out_dev, 0, n_bytes));

    double start = cpu_second();
    convolution(in_host, out_cpu, templ_, nxy);
    double elaps = cpu_second() - start;

    dim3 block(dimx);
    dim3 grid((nxy - 1) / block.x + 1);
    stencil_1d << <grid, block >> > (in_dev, out_dev);
    CHECK(cudaDeviceSynchronize());
    elaps = cpu_second() - start;
    printf("stencil_1d Time elapsed %f sec\n", elaps);
    CHECK(cudaMemcpy(out_gpu, out_dev, n_bytes, cudaMemcpyDeviceToHost));
    check_result(out_cpu, out_gpu, nxy);
    CHECK(cudaMemset(out_dev, 0, n_bytes));

    float* dcoef_ro;
    CHECK(cudaMalloc((void**)&dcoef_ro, TEMP_RADIO_SIZE * sizeof(float)));
    CHECK(cudaMemcpy(dcoef_ro, templ_, TEMP_RADIO_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    stencil_1d_readonly << <grid, block >> > (in_dev, out_dev, dcoef_ro);
    CHECK(cudaDeviceSynchronize());
    elaps = cpu_second() - start;
    printf("stencil_1d_readonly Time elapsed %f sec\n", elaps);
    CHECK(cudaMemcpy(out_gpu, out_dev, n_bytes, cudaMemcpyDeviceToHost));
    check_result(out_cpu, out_gpu, nxy);

    cudaFree(dcoef_ro);
    cudaFree(in_dev);
    cudaFree(out_dev);
    free(out_gpu);
    free(out_cpu);
    free(in_host);
    cudaDeviceReset();

    return 0;
}