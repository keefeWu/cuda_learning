#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <iostream>

 __global__ void kernel(float * d_matrix, size_t pitch, size_t rows, size_t cols) {
    int count = 1;
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < rows; j += blockDim.y * gridDim.y) 
    {
        float* row_d_matrix = (float*)((char*)d_matrix + j*pitch);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cols; i += blockDim.x * gridDim.x)
        {
            row_d_matrix[i] = count;
            count++;
        } 
    }
    // d_matrix[0] = 1;
}

int main(int argc, char **argv)
{
	// device pointers.
	float *d_pitch;
	float *d_normal;
 
	// matrix size.
	size_t cols = 128;
	size_t rows = 16;
	
	size_t pitch = 0;
	
	// alloc the data form gpu memory.
	cudaMallocPitch((void**)&d_pitch, &pitch, cols*sizeof(float), rows);
	cudaMalloc((void**)(&d_normal), rows*cols*sizeof(float));
	
	// test the data address.
	fprintf(stdout, "row size(in bytes) = %.2f*128.\n", pitch/128.0f);
	std::cout<<"d_pitch:"<<d_pitch<<std::endl;
	std::cout<<"d_normal:"<<d_normal<<std::endl;
    std::cout<<"pitch:"<<pitch<<std::endl;
	std::cout<<"occupy_num:"<<pitch/sizeof(float)<<std::endl;
	std::cout<<"sizeof(float):"<<sizeof(float)<<std::endl;

	fprintf(stdout, "the head address of d_pitch  mod 128 = %x.\n", ((long)d_pitch)%128);
	fprintf(stdout, "the head address of d_normal mod 128 = %x.\n", ((long)d_normal)%128);
	


    float *d_matrix;
    float *dc_matrix;
    dc_matrix = (float*)malloc(sizeof(float)* cols * rows);
    cudaMallocPitch(&d_matrix, &pitch, cols*sizeof(float), rows);
    kernel<<<128,128>>>(d_matrix, pitch, rows, cols);
    // cudaMemcpy2D(dc_matrix, cols * sizeof(float), d_matrix, pitch, cols * sizeof(float), rows, cudaMemcpyDeviceToHost);
    cudaMemcpy(dc_matrix, d_matrix, rows*cols * sizeof(float), cudaMemcpyDeviceToHost);
    int count = 0;
    for(int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout<<dc_matrix[count]<<" ";
            ++count;
        }
        std::cout<<std::endl;
    }
    cudaFree(d_matrix);
    free(dc_matrix);

	cudaFree(d_pitch);
	cudaFree(d_normal);
 
	return 0;
}