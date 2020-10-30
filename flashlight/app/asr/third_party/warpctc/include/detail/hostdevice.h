#pragma once

#ifdef __CUDACC__
    #define HOSTDEVICE __host__ __device__
#else
    #define HOSTDEVICE
#endif

// NOTE(dzhwinter)
// the warp primitive is different in cuda9(Volta) GPU.
// add a wrapper to compatible with cuda7 to cuda9
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define DEFAULT_MASK 0u
template<typename T>
__forceinline__ __device__ T __shfl_down(T input, int delta) {
  return __shfl_down_sync(DEFAULT_MASK, input, delta);
}

template<typename T>
__forceinline__ __device__ T __shfl_up(T input, int delta) {
  return __shfl_up_sync(DEFAULT_MASK, input, delta);
}

#endif
