#include <stdio.h>
#include <time.h>

__global__ void add(int* a, int* b, int* c, int num)
{
	int i = threadIdx.x;
	if(i < num)
	{
		c[i] = a[i] + b[i];
	}
}

int main(void)
{
	// init data
	int num = 10;
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


	// get data
	cudaMemcpy(c, c_gpu, num * sizeof(int), cudaMemcpyDeviceToHost);

	// visualization
	for(int i = 0; i < num; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	return 0;
}