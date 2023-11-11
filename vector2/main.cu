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
__global__ void GPUVectorAdd(const GPUVectorDevice<T> v1, const GPUVectorDevice<T> v2, GPUVectorDevice<T> v3)
{
   long int ind = blockIdx.x * blockDim.x + threadIdx.x;
   long int N = v1.size();

   for(; ind < N; ind += blockDim.x*gridDim.x)
      v3[ind] = v1[ind] + v2[ind];
}

int main()
{
   cudaSetupDevice();

   {
      using type = float;
      long int n = 1 << 25;
      Vector<type> v1(n), v2(n), v3(n);
      v1.rand();
      v2.rand();

      GPUVector<type> v1_gpu = ToDevice(v1);
      GPUVector<type> v2_gpu = ToDevice(v2);
      GPUVector<type> v3_gpu(n);

      CPUVectorAdd(v1, v2, v3);

      timer t;
      GPUVectorAdd<type><<< 1024, 512 >>>(v1_gpu.pass(), v2_gpu.pass(), v3_gpu.pass());
      ASSERT_CUDA_SUCCESS( cudaDeviceSynchronize() );
      std::printf("GPU vector add took %.4e seconds\n\n", t.wall_time());

      FromDevice(v3_gpu, v2);
      assert(within_tol_abs(v2, v3));
   }

   cudaDeviceReset();
   return EXIT_SUCCESS;
}

