#include "freshman.h"

__device__ float dev_data;

__global__ void check_global_variable()
{
	printf("Device: The value of the global variable is %f\n", dev_data);
	dev_data += 2.0;
}

int main()
{
	float value = 3.14f;
	cudaMemcpyToSymbol(dev_data, &value, sizeof(float));
	printf("Host: copy %f to the global variable\n", value);
	check_global_variable << <1, 1 >> > ();
	cudaMemcpyFromSymbol(&value, dev_data, sizeof(float));
	printf("Host: the value changed by the kernel to %f \n", value);
	cudaDeviceReset();
	return 0;
}