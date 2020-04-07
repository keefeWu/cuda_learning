#include <stdio.h>

__global__ void sum(float *a, float *b)
{
	int tid = threadIdx.x;

	__shared__ float sData[16];
	sData[tid] = a[tid];
	__syncthreads();
	for(int i = 8; i > 0; i /= 2)
	{
		if(tid < i)
		{
			sData[tid] = sData[tid] + sData[tid + i];
		}
		__syncthreads();
	}
	if(tid == 0)
	{
		b[0] = sData[0];
	}
}

int main()
{
	float a[16];
	for(int i = 0; i < 16; i++)
	{
		a[i] = i*(i+1);
	}
	float *aGpu;
	cudaMalloc((void**)&aGpu, 16 * sizeof(float));
	cudaMemcpy(aGpu, a, 16 * sizeof(float), cudaMemcpyHostToDevice);

	float *bGpu;
	cudaMalloc((void**)&bGpu, 1 * sizeof(float));
	sum<<<1, 16>> >(aGpu, bGpu);

	float b[1];
	cudaMemcpy(b, bGpu, 1 * sizeof(float), cudaMemcpyDeviceToHost);
	printf("b: %f\n",b[0]);
	return 0;
}