#include <stdio.h>
#include <sys/time.h>


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

__global__ void get_hist(float *a, int *hist)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int idx = tid + bid * blockDim.x;

	atomicAdd(&hist[(int)a[idx]], 1);
}

int main()
{
	int size = 32000000;
	float *a = new float[size];

	int length = 10;
	for(int i = 0; i < size; i++)
	{
		a[i] = i*(i+1) % length;
	}

	int hist[length] = {0};

	float *aGpu;
	cudaMalloc((void**)&aGpu, size * sizeof(float));
	cudaMemcpy(aGpu, a, size * sizeof(float), cudaMemcpyHostToDevice);

	int *histGpu;
	cudaMalloc((void**)&histGpu, length * sizeof(int));
	cudaMemcpy(histGpu, hist, length * sizeof(int), cudaMemcpyHostToDevice);

	struct timeval startTime, endTime;
	gettimeofday(&startTime, NULL);
	// get_hist<<<1, size>> >(aGpu, histGpu);
	get_hist<<<size / 512, 512>> >(aGpu, histGpu);
	gettimeofday(&endTime, NULL);
	printf("cuda use time: %d\n",
		(endTime.tv_sec - startTime.tv_sec)*1000000 + (endTime.tv_usec - startTime.tv_usec));
	
	gettimeofday(&startTime, NULL);
	for(int i = 0; i < size; i++)
	{
		hist[(int)a[i]] += 1;
	}
	gettimeofday(&endTime, NULL);
	printf("cpu use time: %d\n",
		(endTime.tv_sec - startTime.tv_sec)*1000000 + (endTime.tv_usec - startTime.tv_usec));
	
	// printf("\ncpu:\n");
	// for(int i = 0; i < length; i++)
	// {
	// 	printf("%.6d ",hist[i]);
	// }

	// cudaMemcpy(hist, histGpu, length * sizeof(int), cudaMemcpyDeviceToHost);
	// printf("\ngpu:\n");
	// for(int i = 0; i < length; i++)
	// {
	// 	printf("%.6d ",hist[i]);
	// }

	return 0;
}