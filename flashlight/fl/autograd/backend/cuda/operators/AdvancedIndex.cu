/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <af/array.h>
#include <cstdio>
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/CudaUtils.h"
#include "flashlight/fl/common/DevicePtr.h"

#define GRID_SIZE 32
#define BLOCK_SIZE 256

template <class Float>
__global__ void gradAdvancedIndexKernel(
    const Float* inp,
    const dim_t* idxStart,
    const dim_t* idxEnd,
    const dim_t* outDims,
    const dim_t* idxArr,
    Float* out) {
  // Compute striding information for
  // the input and output tensors
  dim_t dims[4], strides[4];
  dim_t outStrides[4];
  for (int i = 0; i < 4; i++) {
    dims[i] = idxEnd[i] - idxStart[i];
  }
  strides[0] = 1;
  outStrides[0] = 1;
  // arrayfire dimensions are inverted compared to numpy
  // hence, stride computation starts from 1 to 4
  for (int i = 1; i < 4; i++) {
    strides[i] = strides[i - 1] * dims[i - 1];
    outStrides[i] = outStrides[i - 1] * outDims[i - 1];
  }

  // Map CUDA thread to an element in the input array
  for (dim_t tid = threadIdx.x + blockIdx.x * BLOCK_SIZE;
       tid < (strides[3] * dims[3]);
       tid += (GRID_SIZE * BLOCK_SIZE)) {
    // Compute input array index for CUDA thread
    dim_t index[4];
    dim_t cursor = tid;
    for (int i = 3; i >= 0; i--) {
      index[i] = cursor / strides[i];
      cursor = cursor % strides[i];
    }

    dim_t inpIdx = tid;
    dim_t outIdx = 0;
    for (int i = 0; i < 4; i++) {
      // If indexing array specified, use it
      if (idxArr[i]) {
        auto idxArrPtr = (dim_t*)idxArr[i];
        outIdx += idxArrPtr[index[i]] * outStrides[i];
      } else {
        outIdx += (idxStart[i] + index[i]) * outStrides[i];
      }
    }
    // atomic addition is done to ensure correct
    // gradient computation for repeated indices
    atomicAdd(&out[outIdx], inp[inpIdx]);
  }
}

namespace fl {

void gradAdvancedIndex(
    const Variable& inp,
    const af::dim4& idxStart,
    const af::dim4& idxEnd,
    const af::dim4& outDims,
    const std::vector<af::array>& idxArr,
    Variable& out) {
  auto inpType = inp.type();
  auto outType = out.type();

  if ((inpType != f32) && (inpType != f16)) {
    throw std::invalid_argument("Input type must be f16/f32");
  }
  if ((outType != f32) && (outType != f16)) {
    throw std::invalid_argument("Output type must be f16/f32");
  }
  if (idxArr.size() != 4) {
    throw std::invalid_argument("Index array vector must be length 4");
  }

  DevicePtr idxArrRaw[4];
  void* idxPtr[4];
  // Extract raw device pointers for dimensions
  // that have an array as af::index variable
  for (int i = 0; i < 4; i++) {
    idxPtr[i] = NULL;
    if (!idxArr[i].isempty()) {
      auto idxType = idxArr[i].type();
      if ((idxType != s64) && (idxType != s32)) {
        throw std::invalid_argument("Index type must be s32/s64");
      }
      if (idxType == s32) {
        idxArrRaw[i] = DevicePtr(idxArr[i].as(s64));
      } else {
        idxArrRaw[i] = DevicePtr(idxArr[i]);
      }
      idxPtr[i] = idxArrRaw[i].get();
    }
  }
  if (outType == f16) {
    out = out.as(f32);
  }
  DevicePtr inpRaw(inp.array());
  DevicePtr outRaw(out.array());
  if (inpType == f16) {
    inpRaw = DevicePtr(inp.as(f32).array());
  }

  cudaStream_t stream = cuda::getActiveStream();

  af::array arrIdxStart(4, s64);
  af::array arrIdxEnd(4, s64);
  af::array arrOutDims(4, s64);
  af::array arrIdxPtr(4, s64);
  DevicePtr devIdxStart(arrIdxStart);
  DevicePtr devIdxEnd(arrIdxEnd);
  DevicePtr devOutDims(arrOutDims);
  DevicePtr devIdxPtr(arrIdxPtr);

  // Transformer indexing information to device
  FL_CUDA_CHECK(cudaMemcpyAsync(
      devIdxStart.get(),
      idxStart.get(),
      4 * sizeof(dim_t),
      cudaMemcpyHostToDevice,
      stream));
  FL_CUDA_CHECK(cudaMemcpyAsync(
      devIdxEnd.get(),
      idxEnd.get(),
      4 * sizeof(dim_t),
      cudaMemcpyHostToDevice,
      stream));
  FL_CUDA_CHECK(cudaMemcpyAsync(
      devOutDims.get(),
      outDims.get(),
      4 * sizeof(dim_t),
      cudaMemcpyHostToDevice,
      stream));
  FL_CUDA_CHECK(cudaMemcpyAsync(
      devIdxPtr.get(),
      reinterpret_cast<dim_t*>(idxPtr),
      4 * sizeof(dim_t),
      cudaMemcpyHostToDevice,
      stream));

  gradAdvancedIndexKernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
      static_cast<const float*>(inpRaw.get()),
      static_cast<const dim_t*>(devIdxStart.get()),
      static_cast<const dim_t*>(devIdxEnd.get()),
      static_cast<const dim_t*>(devOutDims.get()),
      static_cast<const dim_t*>(devIdxPtr.get()),
      static_cast<float*>(outRaw.get()));
  FL_CUDA_CHECK(cudaPeekAtLastError());

  if (outType == f16) {
    out = out.as(f16);
  }
}

} // namespace fl
