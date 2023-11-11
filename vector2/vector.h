#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <utility>

#include <cuda_runtime.h>

#define ASSERT_CPU(condition)                                                                                         \
   do {                                                                                                               \
      if(!(condition)) {                                                                                              \
         std::fprintf(stderr, "Error on line %i, file %s: condition %s failed\n", __LINE__, __FILE__, (#condition));  \
         std::exit(EXIT_FAILURE);                                                                                     \
      }                                                                                                               \
   } while(false)

#define ASSERT_CUDA_SUCCESS(cudaCall)                                                                             \
   do {                                                                                                           \
      cudaError_t error = cudaCall;                                                                               \
      if(error != cudaSuccess) {                                                                                  \
         std::fprintf(stderr, "Error on line %i, file %s: %s\n", __LINE__, __FILE__, cudaGetErrorString(error));  \
         std::exit(EXIT_FAILURE);                                                                                 \
      }                                                                                                           \
   } while(false)                                                                                                 \


template<typename T>
class Vector;

template<typename T>
class GPUVector;

template<typename T>
GPUVector<T> ToDevice(const Vector<T> & x) noexcept;

template<typename T>
Vector<T> FromDevice(const GPUVector<T> & x) noexcept;

template<typename T>
void ToDevice(const Vector<T> & x, GPUVector<T> & y) noexcept;

template<typename T>
void FromDevice(const GPUVector<T> & x, Vector<T> & y) noexcept;

template<typename T>
void print(const Vector<T> & v) noexcept;

template<typename T>
struct default_tol
{
   static constexpr T x{100 * std::numeric_limits<T>::epsilon()};
};

template<typename T>
bool within_tol_abs(T val1, T val2, T tol = default_tol<T>::x) noexcept;

template<typename T>
bool within_tol_abs(const Vector<T> & v1, const Vector<T> & v2, T tol = default_tol<T>::x) noexcept;

template<typename T>
bool is_equal(const Vector<T> & v1, const Vector<T> & v2) noexcept;

template<typename T>
class Vector
{
public:

   Vector(long int n) noexcept;
   Vector(const Vector & v) noexcept;
   Vector(Vector && v) noexcept;
   Vector & operator=(const Vector & v) noexcept;
   Vector & operator=(Vector && v) noexcept;
   ~Vector() noexcept;

   void rand() noexcept;
   auto size() const noexcept;

   T & operator[](long int i) noexcept { return data[i]; }
   const T & operator[](long int i) const noexcept { return data[i]; }

   friend GPUVector<T> ToDevice<>(const Vector<T> & x) noexcept;
   friend Vector<T> FromDevice<>(const GPUVector<T> & x) noexcept;
   friend void ToDevice<>(const Vector<T> & x, GPUVector<T> & y) noexcept;
   friend void FromDevice<>(const GPUVector<T> & x, Vector<T> & y) noexcept;

private:
   T* data;
   long int sz;
};


template<typename T>
class GPUVector
{
public:

   GPUVector(long int n) noexcept;
   GPUVector(const GPUVector & v) noexcept;
   GPUVector(GPUVector && v) noexcept;
   GPUVector & operator=(const GPUVector & v) noexcept;
   GPUVector & operator=(GPUVector && v) noexcept;
   ~GPUVector() noexcept;

   T* data();
   const T* data() const;
   auto size() const;

   friend GPUVector<T> ToDevice<>(const Vector<T> & x) noexcept;
   friend Vector<T> FromDevice<>(const GPUVector<T> & x) noexcept;
   friend void ToDevice<>(const Vector<T> & x, GPUVector<T> & y) noexcept;
   friend void FromDevice<>(const GPUVector<T> & x, Vector<T> & y) noexcept;

private:
   T* ptr;
   long int sz;
};


template<typename T>
GPUVector<T>::GPUVector(long int n) noexcept
{
   ASSERT_CPU(n > -1);
   sz = n;
   ASSERT_CUDA_SUCCESS( cudaMalloc((void**) &ptr, sz*sizeof(T)) );
   ASSERT_CUDA_SUCCESS( cudaMemset(ptr, 0x0, sz*sizeof(T)) );
}


template<typename T>
GPUVector<T>::GPUVector(const GPUVector<T> & v) noexcept
{
   sz = v.sz;
   ASSERT_CUDA_SUCCESS( cudaMalloc((void**) &ptr, sz*sizeof(T)) );
   ASSERT_CUDA_SUCCESS( cudaMemcpy(ptr, v.ptr, sz*sizeof(T), cudaMemcpyDeviceToDevice) );
}


template<typename T>
GPUVector<T>::GPUVector(GPUVector<T> && v) noexcept
{
   sz = std::exchange(v.sz, 0);
   ptr = std::exchange(v.ptr, nullptr);
}


template<typename T>
GPUVector<T> & GPUVector<T>::operator=(const GPUVector<T> & v) noexcept
{
   ASSERT_CPU(sz == v.sz);
   ASSERT_CUDA_SUCCESS( cudaMemcpy(ptr, v.ptr, sz*sizeof(T), cudaMemcpyDeviceToDevice) );
   return *this;
}


template<typename T>
GPUVector<T> & GPUVector<T>::operator=(GPUVector<T> && v) noexcept
{
   ASSERT_CPU(sz == v.sz);
   ASSERT_CUDA_SUCCESS( cudaFree(ptr) );

   sz = std::exchange(v.sz, 0);
   ptr = std::exchange(v.ptr, nullptr);
   return *this;
}


template<typename T>
GPUVector<T>::~GPUVector<T>() noexcept
{
   if(ptr)
      ASSERT_CUDA_SUCCESS( cudaFree(ptr) );
}


template<typename T>
T* GPUVector<T>::data()
{
   return ptr;
}


template<typename T>
const T* GPUVector<T>::data() const
{
   return ptr;
}


template<typename T>
auto GPUVector<T>::size() const
{
   return sz;
}


template<typename T>
Vector<T>::Vector(long int n) noexcept
{
   ASSERT_CPU(n > -1);
   sz = n;
   ASSERT_CPU( data = (T*)std::calloc(sz, sizeof(T)) );
}


template<typename T>
Vector<T>::Vector(const Vector<T> & v) noexcept
{
   sz = v.sz;
   ASSERT_CPU( data = (T*)std::malloc(sz*sizeof(T)) );
   std::memcpy(data, v.data, sz*sizeof(T));
}


template<typename T>
Vector<T>::Vector(Vector<T> && v) noexcept
{
   sz = std::exchange(v.sz, 0);
   data = std::exchange(v.data, nullptr);
}


template<typename T>
Vector<T> & Vector<T>::operator=(const Vector<T> & v) noexcept
{
   ASSERT_CPU(sz == v.sz);
   std::memcpy(data, v.data, sz*sizeof(T));
   return *this;
}


template<typename T>
Vector<T> & Vector<T>::operator=(Vector<T> && v) noexcept
{
   ASSERT_CPU(sz == v.sz);
   std::free(data);

   sz = std::exchange(v.sz, 0);
   data = std::exchange(v.data, nullptr);
   return *this;
}


template<typename T>
Vector<T>::~Vector() noexcept
{
   if(data)
      std::free(data);
}


template<typename T>
void Vector<T>::rand() noexcept
{
   using namespace std::chrono;
   microseconds ms = duration_cast<microseconds> (system_clock::now().time_since_epoch());
   std::srand(static_cast<unsigned>(std::time(0)) + static_cast<unsigned>(ms.count()));
   for(auto i = 0L; i < sz; ++i)
      data[i] = T(std::rand())/T(RAND_MAX);
}


template<typename T>
auto Vector<T>::size() const noexcept
{
   return sz;
}


template<typename T>
GPUVector<T> ToDevice(const Vector<T> & x) noexcept
{
   GPUVector<T> y(x.sz);
   ToDevice(x, y);
   return y;
}


template<typename T>
Vector<T> FromDevice(const GPUVector<T> & x) noexcept
{
   Vector<T> y(x.sz);
   FromDevice(x, y);
   return y;
}


template<typename T>
void ToDevice(const Vector<T> & x, GPUVector<T> & y) noexcept
{
   ASSERT_CPU( x.sz == y.sz );
   ASSERT_CUDA_SUCCESS( cudaMemcpy(y.ptr, x.data, x.sz*sizeof(T), cudaMemcpyHostToDevice) );
}


template<typename T>
void FromDevice(const GPUVector<T> & x, Vector<T> & y) noexcept
{
   ASSERT_CPU( x.sz == y.sz );
   ASSERT_CUDA_SUCCESS( cudaMemcpy(y.data, x.ptr, x.sz*sizeof(T), cudaMemcpyDeviceToHost) );
}

template<typename T>
void print(const Vector<T> & v) noexcept
{
   for(auto i = 0L; i < v.size(); ++i)
      std::cout << v[i] << '\n';
}

template<typename T>
bool within_tol_abs(T val1, T val2, T tol) noexcept
{
   T abs_val = val1 > val2 ? (val1 - val2) : (val2 - val1);
   return abs_val <= tol;
}

template<typename T>
bool within_tol_abs(const Vector<T> & v1, const Vector<T> & v2, T tol) noexcept
{
   ASSERT_CPU(v1.size() == v2.size());
   for(auto i = 0L; i < v1.size(); ++i)
      if(!within_tol_abs(v1[i], v2[i], tol))
         return false;
   return true;
}

template<typename T>
bool is_equal(const Vector<T> & v1, const Vector<T> & v2) noexcept
{
   ASSERT_CPU(v1.size() == v2.size());
   for(auto i = 0L; i < v1.size(); ++i)
      if(v1[i] != v2[i])
         return false;
   return true;
}

