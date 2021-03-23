#include <stdio.h>
#include <ctime>
#include <string> 
#include <iostream>   
#include <stdlib.h>
typedef bool TYPE;
bool single_adder(TYPE A, TYPE B, TYPE& Ci, TYPE& Si)
{
    Ci = A & B; // 与门表示进位
    Si = A ^ B; // 异或门表示和
    printf("%d + %d = %d, Ci: %d\n",A,B,Si,Ci);
    return 1;
}
bool multi_adder(TYPE A, TYPE B, TYPE Ci0, TYPE& Ci, TYPE& Si)
{
    Si = A ^ B ^ Ci0; // 异或门表示和
    Ci = A & B; // 进位产生信号
    Ci = ((A ^ B) & Ci0) | Ci; // 进位传递信号
    // Ci = ((A | B) & Ci0) | Ci; // 直接考虑AB其中一个是1且传递的进位符也是1的情况决定是否进位
    printf("%d + %d + %d = %d, Ci: %d\n",A,B,Ci0,Si,Ci);
    return 1;
}

__global__ void full_adder(TYPE *num1, TYPE *num2, TYPE *result, TYPE* Ci0_, int startIdx, int length)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index >= length)
	{
		return;
	}
	TYPE Ci0 = Ci0_[0];
	TYPE Ci = 0;
	for(int i = 0; i < index; i++)
	{
		TYPE A = num1[i + startIdx];
		TYPE B = num2[i + startIdx];
		Ci = A & B;
		Ci = ((A ^ B) & Ci0) | Ci; // 进位传递信号
		Ci0 = Ci;
	}
	int i = index;
	TYPE A = num1[i + startIdx];
	TYPE B = num2[i + startIdx];
	TYPE Si = A ^ B ^ Ci0; // 异或门表示和
	result[index + startIdx] = Si;
	if(index == length - 1)
	{
		Ci = A & B;
		Ci = ((A ^ B) & Ci0) | Ci; // 进位传递信号
		Ci0_[0] = Ci;
	}
}
int convert_string_to_array(std::string num1, std::string num2, TYPE* &A, TYPE* &B)
{
	    // 如果他俩不一样长就给短的补0
		int zero_num = num1.length() - num2.length();
		std::string* temp;
		if(zero_num > 0)
		{
			temp = &num2;
		}
		else if(zero_num < 0)
		{
			temp = &num1;
		}                                                   
		for(int i = 0; i < abs(zero_num); i++)
		{
			*temp = "0" + *temp;
		}
		int length = num1.length();
		A = new TYPE[length];
		B = new TYPE[length];
		for(int i = 0; i < length; i++)
		{
			A[i] = num1[length - 1 - i] - '0';
			B[i] = num2[length - 1 - i] - '0';
		}
		return length;
}
int main()
{
    std::string num1 = "10101";
    std::string num2 = "11111";
	TYPE *ACpu;
	TYPE *BCpu;
	int length = convert_string_to_array(num1, num2, ACpu, BCpu);
	for(int i = 0; i < length; i++)
	{
		printf("%d ", ACpu[i]);
	}
	printf("\n");

	for(int i = 0; i < length; i++)
	{
		printf("%d ", BCpu[i]);
	}
	printf("\n");

	TYPE *AGpu;
	cudaMalloc((void**)&AGpu, length * sizeof(TYPE));
	cudaMemcpy(AGpu, ACpu, length * sizeof(TYPE), cudaMemcpyHostToDevice);
	TYPE *BGpu;
	cudaMalloc((void**)&BGpu, length * sizeof(TYPE));
	cudaMemcpy(BGpu, BCpu, length * sizeof(TYPE), cudaMemcpyHostToDevice);

	TYPE *resultGpu;
	cudaMalloc((void**)&resultGpu, length * sizeof(TYPE));

	TYPE *CiCpu = new TYPE[1];
	CiCpu = 0;
	TYPE *CiGpu;
	cudaMalloc((void**)&CiGpu, 1 * sizeof(TYPE));
	cudaMemcpy(CiCpu, CiGpu, 1 * sizeof(TYPE), cudaMemcpyHostToDevice);

	
	int threadNum = 1;
	int blockNum = 4;
	int totalNum = threadNum * blockNum;
	for(int i = 0; totalNum * i < length; i++)
	{
		printf("i: %d, i2: %d\n", totalNum * i, min(length - totalNum * i, totalNum));
		full_adder<<<blockNum, threadNum>> >(AGpu, BGpu, resultGpu, CiGpu, totalNum * i, min(length - totalNum * i, totalNum));
	} 

	TYPE *result = new TYPE[length];
	cudaMemcpy(result, resultGpu, length * sizeof(TYPE), cudaMemcpyDeviceToHost);
	for(int i = 0; i < length; i++)
	{
		printf("%d ", result[length - 1 - i]);
	}
	printf("\n");
	return 0;
}