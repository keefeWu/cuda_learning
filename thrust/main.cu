#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <vector>
struct saxpy_functor
{
  const float a;

  saxpy_functor(float _a) : a(_a) {}

  __host__ __device__
  float operator()(const float& x, const float& y) const
  { 
    return y * x + a;
  }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
  // Y <- A * X + Y
  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
  thrust::device_vector<float> temp(X.size());
   
  // temp <- A
  thrust::fill(temp.begin(), temp.end(), A);
    
  // temp <- A * X
  thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(), thrust::multiplies<float>());

  // Y <- A * X + Y
  thrust::transform(temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<float>());
}

int main(void)
{
	thrust::host_vector<float> H(4);
	std::vector<float> I(4);

	// initialize individual elements
	H[0] = 14;
	H[1] = 20;
	H[2] = 38;
	H[3] = 46;
	  // H.size() returns the size of vector H
	std::cout << "H has size " << H.size() << std::endl;

	// print contents of H
	for(int i = 0; i < H.size(); i++)
	{
		std::cout << "H[" << i << "] = " << H[i] << std::endl;
	}
	// Copy host_vector H to device_vector D
	thrust::device_vector<float> D = H;
	// print contents of D
	for(int i = 0; i < D.size(); i++)
	{
		std::cout << "D[" << i << "] = " << D[i] << std::endl;
	}
	// elements of D can be modified
	D[0] = 99;
	D[1] = 88;
	std::cout<< "D values are changed!" << std::endl;
	for(int i = 0; i < D.size(); i++)
	{
		std::cout << "D[" << i << "] = " << D[i] << std::endl;
	}
	// print contents of H
	for(int i = 0; i < H.size(); i++)
	{
		std::cout << "H[" << i << "] = " << H[i] << std::endl;
	}
	// resize H
  	H.resize(5);
	std::cout << "H now has size " << H.size() << std::endl;
	// set the elements of H to 0, 1, 2, 3, ...
	thrust::sequence(H.begin(), H.end());
		// print contents of H
	for(int i = 0; i < H.size(); i++)
	{
		std::cout << "H[" << i << "] = " << H[i] << std::endl;
	}
	thrust::fill(D.begin(), D.begin() + 3, 9);
	for(int i = 0; i < D.size(); i++)
	{
		std::cout << "D[" << i << "] = " << D[i] << std::endl;
	}
	thrust::device_vector<float> Y(4);
	// compute Y = -X
  	thrust::transform(D.begin(), D.end(), Y.begin(), thrust::negate<int>());
	for(int i = 0; i < Y.size(); i++)
	{
		std::cout << "Y[" << i << "] = " << Y[i] << std::endl;
	}

	saxpy_fast(2.5, D, Y);
	for(int i = 0; i < Y.size(); i++)
	{
		std::cout << "Y[" << i << "] = " << Y[i] << std::endl;
	}
	return 0;

}