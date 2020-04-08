#include <stdio.h>

#define LENGTH 16
#define BLOCKNUM 2
#define THREADNUM 4
__global__ void dot_product(float *a, float *b, float* r)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int total_thread_num = THREADNUM * BLOCKNUM;

	__shared__ float sData[THREADNUM];
	int global_id = tid + bid * THREADNUM;
	sData[tid] = 0;
	while(global_id < LENGTH)
	{
		sData[tid] += a[global_id] * b[global_id];
		global_id += total_thread_num;
	}
	__syncthreads();
	for(int i = THREADNUM/2; i > 0; i /= 2)
	{
		if(tid < i)
		{
			sData[tid] = sData[tid] + sData[tid + i];
		}
		__syncthreads();
	}
	if(tid == 0)
	{
		r[bid] = sData[0];
	}
}

int main()
{
	float a[LENGTH];
	float b[LENGTH];
	for(int i = 0; i < LENGTH; i++)
	{
		a[i] = i*(i+1);
		b[i] = i*(i-2);
	}
	float *aGpu;
	cudaMalloc((void**)&aGpu, LENGTH * sizeof(float));
	cudaMemcpy(aGpu, a, LENGTH * sizeof(float), cudaMemcpyHostToDevice);

	float *bGpu;
	cudaMalloc((void**)&bGpu, LENGTH * sizeof(float));
	cudaMemcpy(bGpu, b, LENGTH * sizeof(float), cudaMemcpyHostToDevice);

	float *rGpu;
	cudaMalloc((void**)&rGpu, BLOCKNUM * sizeof(float));
	dot_product<<<BLOCKNUM, LENGTH>> >(aGpu, bGpu, rGpu);

	float r[BLOCKNUM];
	cudaMemcpy(r, rGpu, BLOCKNUM * sizeof(float), cudaMemcpyDeviceToHost);

	float result = 0;
	for(int i = 0; i < BLOCKNUM; i++)
	{
		result += r[i];
	}
	printf("result: %f\n",result);
	return 0;
}