#include <stdio.h>
#include <stdlib.h>
#include "get_time.h"

#include "vector.h"

int main(void)
{
   printf("\n");
   int dev = 0;
   cudaDeviceProp deviceProp;
   cudaError_t  devError = cudaGetDeviceProperties(&deviceProp, dev);
   if(devError != cudaSuccess) {
      fprintf(stderr, "Line %i: No GPU device available\n", __LINE__);
      exit(EXIT_FAILURE);
   }

   const long int n = 2 << 25;
   const double c1 = 0.5, c2 = 0.75;
   Vector v1 = VectorInitRandom(n);
   Vector v2 = VectorInitRandom(n);
   Vector v3 = VectorInit(n);

   GPUVector v1_gpu = GPUVectorCopyInit(v1);
   GPUVector v2_gpu = GPUVectorCopyInit(v2);
   GPUVector v3_gpu = GPUVectorInit(n);

   TIME( VectorAddCPU(c1, c2, v1, v2, v3) );

   const dim3 block(256);
   const dim3 grid((n+block.x-1)/block.x);

   double start_gpu = get_cur_time();
   VectorAddGPU<<< grid, block >>>(c1, c2, v1_gpu, v2_gpu, v3_gpu);
   cudaDeviceSynchronize();
   printf("vector add on GPU took: %.4e seconds\n", get_cur_time() - start_gpu);

   double diff = VectorMaxDifference(v3, v3_gpu);
   printf("maximum vector difference = %.4e\n", diff);

   VectorFree(v1);
   VectorFree(v2);
   VectorFree(v3);
   GPUVectorFree(v1_gpu);
   GPUVectorFree(v2_gpu);
   GPUVectorFree(v3_gpu);

   cudaDeviceReset();

   return EXIT_SUCCESS;
}


