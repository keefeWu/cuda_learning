#include <stdio.h>
#include <time.h>
#include <sys/time.h>

__global__ void sum(float *a, float *b)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int threadNum = blockDim.x;

	__shared__ float sData[512];
	sData[tid] = a[bid * threadNum + tid];
	__syncthreads();
	for(int i = threadNum / 2; i > 0; i /= 2)
	{
		if(tid < i)
		{
			sData[tid] = sData[tid] + sData[tid + i];
		}
		__syncthreads();
	}
	if(tid == 0)
	{
		b[bid] = sData[0];
	}
}

void cpuSum(float *a, float *b, int sumNum)
{
	for(int j = 0; j < sumNum; j++)
	{
		b[j] = 0;
		for(int i = 0; i < sumNum; i++)
		{
			b[j] += a[i];
		}
	}
}

__global__ void add(int* a, int* b, int* c, int num)
{
	int i;
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int threadNum = blockDim.x;
	i = bid * threadNum + tid;
	if(i < num)
	{
		c[i] = a[i] + b[i];
	}
}

int testSum()
{
	int sumNum = 512;
	int threadNum = 1;
	int blockNum = 512;
	float a[sumNum];
	for(int i = 0; i < sumNum; i++)
	{
		a[i] = i*(i+1);
	}
	float *aGpu;
	cudaMalloc((void**)&aGpu, sumNum * sizeof(float));
	cudaMemcpy(aGpu, a, sumNum * sizeof(float), cudaMemcpyHostToDevice);

	float *bGpu;
	cudaMalloc((void**)&bGpu, sumNum * sizeof(float));
	struct timeval startTime, endTime;
	gettimeofday(&startTime, NULL);
	int loopNum = 1000;
	for(int i = 0; i < loopNum; i++)
	{
		sum<<<blockNum, threadNum>> >(aGpu, bGpu);
	}
	// sum<<<sumNum, sumNum>> >(aGpu, bGpu);
	gettimeofday(&endTime, NULL);
	printf("cuda use time: %d\n",
		(endTime.tv_sec - startTime.tv_sec)*1000000 + (endTime.tv_usec - startTime.tv_usec));

	float b[sumNum];
	cudaMemcpy(b, bGpu, sumNum * sizeof(float), cudaMemcpyDeviceToHost);
	// printf("b: %f\n",b[0]);

	gettimeofday(&startTime, NULL);
	for(int i = 0; i < loopNum; i++)
	{
		cpuSum(a, b, sumNum);
	}
	gettimeofday(&endTime, NULL);
	printf("cpu use time: %d\n",
		(endTime.tv_sec - startTime.tv_sec)*1000000 + (endTime.tv_usec - startTime.tv_usec));


	return 0;
}

int testAdd(void)
{
	// init data
	int num = 5120;
	int threadNum = 128;
	int blockNum = 40;
	int a[num], b[num], c[num];
	int *a_gpu, *b_gpu, *c_gpu;

	for(int i = 0; i < num; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}

	cudaMalloc((void **)&a_gpu, num * sizeof(int));
	cudaMalloc((void **)&b_gpu, num * sizeof(int));
	cudaMalloc((void **)&c_gpu, num * sizeof(int));

	// copy data
	cudaMemcpy(a_gpu, a, num * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(b_gpu, b, num * sizeof(int), cudaMemcpyHostToDevice);

	struct timeval startTime, endTime;
	gettimeofday(&startTime, NULL);
	int loopNum = 10000;
	for(int i = 0; i < loopNum; i++)
	{
		add<<<blockNum, threadNum>> >(a_gpu, b_gpu, c_gpu, num);
	}
	gettimeofday(&endTime, NULL);
	printf("cuda use time: %d\n",
		(endTime.tv_sec - startTime.tv_sec)*1000000 + (endTime.tv_usec - startTime.tv_usec));


	// get data
	cudaMemcpy(c, c_gpu, num * sizeof(int), cudaMemcpyDeviceToHost);

	// // visualization
	// for(int i = 0; i < num; i++)
	// {
	// 	printf("%d + %d = %d\n", a[i], b[i], c[i]);
	// }

	return 0;
}

int main()
{
	testAdd();
	return 0;
}