#pragma once

#include <stdio.h>

#define ASSERT_CUDA_SUCCESS(cudaCall)                                                            \
{                                                                                                \
   cudaError_t error = cudaCall;                                                                 \
   if(error != cudaSuccess) {                                                                    \
      printf("Error on line %i, file %s: %s\n", __LINE__, __FILE__, cudaGetErrorString(error));  \
      exit(EXIT_FAILURE);                                                                        \
   }                                                                                             \
}

typedef struct
{
   double* id;
   long int len;
   size_t nbytes;
} Vector;

typedef struct
{
   double* id;
   long int len;
   size_t nbytes;
} GPUVector;

void VectorAddCPU(double c1, double c2, Vector v1, Vector v2, Vector v3);
__global__ void VectorAddGPU(double c1, double c2, GPUVector v1, GPUVector v2, GPUVector v3);

Vector VectorInit(long int n);
Vector VectorInitRandom(long int n);
void VectorFree(Vector v);

GPUVector GPUVectorInit(long int n);
GPUVector GPUVectorCopyInit(Vector v_cpu);
void GPUVectorFree(GPUVector v);

double VectorMaxDifference(Vector v_cpu, GPUVector v_gpu);
void AssignVectorFromGPU(GPUVector vec_gpu, Vector vec_cpu);
void AssignVectorToGPU(Vector vec_cpu, GPUVector vec_gpu);

