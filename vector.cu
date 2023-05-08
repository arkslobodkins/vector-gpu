#include <assert.h>
#include "vector.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////
void VectorAddCPU(double c1, double c2, Vector v1, Vector v2, Vector v3)
{
   assert(v1.len == v2.len);
   assert(v2.len == v3.len);
   for(long int i = 0; i < v1.len; ++i) {
      v3.id[i] = c1*v1.id[i] + c2*v2.id[i];
   }
}

__global__ void VectorAddGPU(double c1, double c2, GPUVector v1, GPUVector v2, GPUVector v3)
{
   long int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if(tid < v1.len) {
      v3.id[tid] = c1*v1.id[tid] + c2*v2.id[tid];
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
Vector VectorInit(long int n)
{
   assert(n > -1);
   Vector v;
   v.id = (double*) calloc(n, sizeof(double));
   v.len = n;
   v.nbytes = n*sizeof(double);
   return v;
}

Vector VectorInitRandom(long int n)
{
   Vector v = VectorInit(n);
   for(long int i = 0; i < n; ++i) {
      v.id[i] = (double)rand() / (double)RAND_MAX;
   }
   return v;
}

void VectorFree(Vector v)
{
   free(v.id);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
GPUVector GPUVectorInit(long int n)
{
   assert(n > -1);
   GPUVector v;
   ASSERT_CUDA_SUCCESS( cudaMalloc((void **)&v.id, n*sizeof(double)) );
   ASSERT_CUDA_SUCCESS( cudaMemset(v.id, 0, n*sizeof(double)) );
   v.len = n;
   v.nbytes = n*sizeof(double);
   return v;
}

GPUVector GPUVectorCopyInit(Vector vec_cpu)
{
   long int len = vec_cpu.len;
   size_t nbytes = vec_cpu.nbytes;

   GPUVector vec_gpu;
   ASSERT_CUDA_SUCCESS( cudaMalloc((void **)&vec_gpu.id, nbytes ) );
   ASSERT_CUDA_SUCCESS( cudaMemcpy(vec_gpu.id, vec_cpu.id, nbytes, cudaMemcpyHostToDevice ) );
   vec_gpu.len = len;
   vec_gpu.nbytes = nbytes;

   return vec_gpu;
}

void GPUVectorFree(GPUVector v)
{
   ASSERT_CUDA_SUCCESS( cudaFree(v.id) );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
double VectorMaxDifference(Vector vec_cpu, GPUVector vec_gpu)
{
   assert(vec_gpu.len == vec_cpu.len);
   assert(vec_gpu.len > 0);

   Vector v_cpu2 = VectorInit(vec_cpu.len);
   CopyVectorFromGPU(vec_gpu, v_cpu2);

   double max = fabs(vec_cpu.id[0] - v_cpu2.id[0]);
   for(long int i = 1; i < vec_cpu.len; ++i) {
      double diff = fabs(vec_cpu.id[i] - v_cpu2.id[i]);
      if(diff > max)  {
         max = diff;
      }
   }

   VectorFree(v_cpu2);
   return max;
}

void CopyVectorFromGPU(GPUVector vec_gpu, Vector vec_cpu)
{
   assert(vec_gpu.len == vec_cpu.len);
   ASSERT_CUDA_SUCCESS( cudaMemcpy(vec_cpu.id, vec_gpu.id, vec_cpu.nbytes, cudaMemcpyDeviceToHost) );
}

void CopyVectorToGPU(Vector vec_cpu, GPUVector vec_gpu)
{
   assert(vec_cpu.len == vec_gpu.len);
   ASSERT_CUDA_SUCCESS( cudaMemcpy(vec_gpu.id, vec_cpu.id, vec_cpu.nbytes, cudaMemcpyHostToDevice) );
}

