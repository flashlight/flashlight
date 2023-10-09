/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <af/array.h>
#include <af/cuda.h>
#include <af/device.h>
#include <af/dim4.hpp>

#include <stdexcept>
#include <unordered_set>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/Types.h"

#define GRID_SIZE 32
#define BLOCK_SIZE 256

const std::unordered_set<af::dtype> validIndexTypes{
    af::dtype::s32,
    af::dtype::s64,
    af::dtype::u32,
    af::dtype::u64};

template <class Float, class Index>
__global__ void advancedIndexKernel(
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
        auto idxArrPtr = (Index*)idxArr[i];
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
namespace detail {

void advancedIndex(
    const af::array& inp,
    const af::dim4& idxStart,
    const af::dim4& idxEnd,
    const af::dim4& outDims,
    const std::vector<af::array>& idxArr,
    af::array& out) {
  auto inpType = inp.type();
  auto outType = out.type();

  if ((inpType != af::dtype::f32) && (inpType != af::dtype::f16)) {
    throw std::invalid_argument("Input type must be f16/f32");
  }
  if ((outType != af::dtype::f32) && (outType != af::dtype::f16)) {
    throw std::invalid_argument("Output type must be f16/f32");
  }
  if (idxArr.size() != 4) {
    throw std::invalid_argument("Index array vector must be length 4");
  }

  af::dim4 idxPtr;
  // Extract raw device pointers for dimensions
  // that have an array as af::index variable

  // Dtype checking
  std::vector<af::dtype> idxTypes;
  for (int i = 0; i < 4; i++) {
    if (idxArr[i].isempty()) {
      idxPtr[i] = 0;
      continue;
    }
    if (validIndexTypes.find(idxArr[i].type()) == validIndexTypes.end()) {
      throw std::invalid_argument(
          "Index type must be one of s32/s64/u32/u64, observed type is " +
          std::to_string(idxArr[i].type()));
    }
    idxTypes.push_back(idxArr[i].type());
    idxPtr[i] = (dim_t)(idxArr[i].device<void>());
  }
  for (int i = 0; i + 1 < idxTypes.size(); i++) {
    if (idxTypes[i] != idxTypes[i + 1]) {
      throw std::invalid_argument(
          "Index type must be the same across all dimensions");
    }
  }

  af::array inpCast = inp;
  af::array outCast = out;
  if (inpType == af::dtype::f16) {
    inpCast = inp.as(af::dtype::f32);
  }
  if (outType == af::dtype::f16) {
    outCast = out.as(af::dtype::f32);
  }

  void* inpRawPtr = inpCast.device<void>();
  void* outRawPtr = outCast.device<void>();
  af::array arrIdxPtr(4, idxPtr.get());
  af::array arrIdxEnd(4, idxEnd.get());
  af::array arrIdxStart(4, idxStart.get());
  af::array arrOutDims(4, outDims.get());
  void* arrIdxStartDev = arrIdxStart.device<void>();
  void* arrIdxEndDev = arrIdxEnd.device<void>();
  void* arrOutDimsDev = arrOutDims.device<void>();
  void* arrIdxPtrDev = arrIdxPtr.device<void>();

  cudaStream_t stream = afcu::getStream(af::getDevice());
  if (idxTypes.size() == 0 || idxTypes[0] == af::dtype::s32) {
    advancedIndexKernel<float, int32_t><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
        static_cast<const float*>(inpRawPtr),
        static_cast<const dim_t*>(arrIdxStartDev),
        static_cast<const dim_t*>(arrIdxEndDev),
        static_cast<const dim_t*>(arrOutDimsDev),
        static_cast<const dim_t*>(arrIdxPtrDev),
        static_cast<float*>(outRawPtr));
  } else if (idxTypes[0] == af::dtype::s64) {
    advancedIndexKernel<float, int64_t><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
        static_cast<const float*>(inpRawPtr),
        static_cast<const dim_t*>(arrIdxStartDev),
        static_cast<const dim_t*>(arrIdxEndDev),
        static_cast<const dim_t*>(arrOutDimsDev),
        static_cast<const dim_t*>(arrIdxPtrDev),
        static_cast<float*>(outRawPtr));
  } else if (idxTypes[0] == af::dtype::u32) {
    advancedIndexKernel<float, uint32_t><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
        static_cast<const float*>(inpRawPtr),
        static_cast<const dim_t*>(arrIdxStartDev),
        static_cast<const dim_t*>(arrIdxEndDev),
        static_cast<const dim_t*>(arrOutDimsDev),
        static_cast<const dim_t*>(arrIdxPtrDev),
        static_cast<float*>(outRawPtr));
  } else if (idxTypes[0] == af::dtype::u64) {
    advancedIndexKernel<float, uint64_t><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
        static_cast<const float*>(inpRawPtr),
        static_cast<const dim_t*>(arrIdxStartDev),
        static_cast<const dim_t*>(arrIdxEndDev),
        static_cast<const dim_t*>(arrOutDimsDev),
        static_cast<const dim_t*>(arrIdxPtrDev),
        static_cast<float*>(outRawPtr));
  } else {
    throw std::invalid_argument("Index type must be one of s32/s64/u32/u64");
  }
  if (cudaPeekAtLastError() != cudaSuccess) {
    throw std::runtime_error(
        "ArrayFireTensor advancedIndex kernel CUDA failure");
  }

  inpCast.unlock();
  outCast.unlock();
  arrIdxStart.unlock();
  arrIdxEnd.unlock();
  arrOutDims.unlock();
  arrIdxPtr.unlock();
  for (const auto& arr : idxArr) {
    arr.unlock();
  }

  out = outCast;
  if (outType == af::dtype::f16) {
    out = outCast.as(af::dtype::f16);
  }
}

} // namespace detail
} // namespace fl
