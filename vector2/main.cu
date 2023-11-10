#include <cstdlib>
#include <cstdio>
#include <cassert>

#include "timer.h"
#include "vector.h"

#include <cuda_runtime.h>

void cudaSetupDevice()
{
   int device_count = 0;
   cudaGetDeviceCount(&device_count);
   if(device_count == 0) {
      std::fprintf(stderr, "Error: CUDA device not found\n");
      std::exit(EXIT_FAILURE);
   }

   int dev = 0;
   ASSERT_CUDA_SUCCESS( cudaSetDevice(dev) );
}

template<typename T>
void CPUVectorAdd(const Vector<T> & v1, const Vector<T> & v2, Vector<T> & v3)
{
   for(auto i = 0L; i < v1.size(); ++i)
      v3[i] = v1[i] + v2[i];
}

template<typename T>
__global__ void GPUVectorAdd(const GPUVector<T> & v1, const GPUVector<T> & v2, GPUVector<T> & v3)
{
   long int ind = blockIdx.x * blockDim.x + threadIdx.x;
   long int N = v1.size();
   const T* v1_ptr = v1.data();
   const T* v2_ptr = v2.data();
   T* v3_ptr = v3.data();

   for(; ind < N; ind += blockDim.x*gridDim.x)
      v3_ptr[ind] = v1_ptr[ind] + v2_ptr[ind];
}

int main()
{
   cudaSetupDevice();

   int n = 1'000'000;
   Vector<float> v1(n), v2(n), v3(n);
   v1.rand();
   v2.rand();

   GPUVector<float> v1_gpu = ToDevice(v1);
   GPUVector<float> v2_gpu = ToDevice(v2);
   GPUVector<float> v3_gpu(n);

   CPUVectorAdd(v1, v2, v3);

   timer t_cross{};
   GPUVectorAdd<<< 512, 256 >>>(v1_gpu, v2_gpu, v3_gpu);
   ASSERT_CUDA_SUCCESS( cudaDeviceSynchronize() );
   std::printf("GPUVectorAdd on GPU took: %.4e seconds\n\n", t_cross.wall_time());

   FromDevice(v3_gpu, v2);
   assert(within_tol_abs(v2, v3));

   cudaDeviceReset();
   return EXIT_SUCCESS;
}

