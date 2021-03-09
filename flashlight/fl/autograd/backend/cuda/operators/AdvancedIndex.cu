/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <af/array.h>

#include <stdexcept>

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/CppBackports.h"
#include "flashlight/fl/common/backend/cuda/CudaUtils.h"
#include "flashlight/fl/common/DevicePtr.h"

#define GRID_SIZE 32
#define BLOCK_SIZE 256

const fl::cpp::fl_unordered_set<af::dtype> validIndexTypes{s32, s64, u32, u64};

template <class Float, class Index>
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
    idxArrRaw[i] = DevicePtr(idxArr[i]);
    idxPtr[i] = (dim_t)(idxArrRaw[i].get());
  }
  for (int i = 0; i + 1 < idxTypes.size(); i++) {
    if (idxTypes[i] != idxTypes[i + 1]) {
      throw std::invalid_argument(
          "Index type must be the same across all dimensions");
    }
  }
  Variable inpCast = inp;
  if (inpType == f16) {
    inpCast = inp.as(f32);
  }
  if (outType == f16) {
    out = out.as(f32);
  }
  DevicePtr inpRaw(inpCast.array());
  DevicePtr outRaw(out.array());

  af::array arrIdxPtr(4, idxPtr.get());
  af::array arrIdxEnd(4, idxEnd.get());
  af::array arrIdxStart(4, idxStart.get());
  af::array arrOutDims(4, outDims.get());
  DevicePtr devIdxStart(arrIdxStart);
  DevicePtr devIdxEnd(arrIdxEnd);
  DevicePtr devOutDims(arrOutDims);
  DevicePtr devIdxPtr(arrIdxPtr);

  cudaStream_t stream = cuda::getActiveStream();
  if (idxTypes.size() == 0 || idxTypes[0] == s32) {
    gradAdvancedIndexKernel<float, int32_t>
        <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
            static_cast<const float*>(inpRaw.get()),
            static_cast<const dim_t*>(devIdxStart.get()),
            static_cast<const dim_t*>(devIdxEnd.get()),
            static_cast<const dim_t*>(devOutDims.get()),
            static_cast<const dim_t*>(devIdxPtr.get()),
            static_cast<float*>(outRaw.get()));
  } else if (idxTypes[0] == s64) {
    gradAdvancedIndexKernel<float, int64_t>
        <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
            static_cast<const float*>(inpRaw.get()),
            static_cast<const dim_t*>(devIdxStart.get()),
            static_cast<const dim_t*>(devIdxEnd.get()),
            static_cast<const dim_t*>(devOutDims.get()),
            static_cast<const dim_t*>(devIdxPtr.get()),
            static_cast<float*>(outRaw.get()));
  } else if (idxTypes[0] == u32) {
    gradAdvancedIndexKernel<float, uint32_t>
        <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
            static_cast<const float*>(inpRaw.get()),
            static_cast<const dim_t*>(devIdxStart.get()),
            static_cast<const dim_t*>(devIdxEnd.get()),
            static_cast<const dim_t*>(devOutDims.get()),
            static_cast<const dim_t*>(devIdxPtr.get()),
            static_cast<float*>(outRaw.get()));
  } else if (idxTypes[0] == u64) {
    gradAdvancedIndexKernel<float, uint64_t>
        <<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
            static_cast<const float*>(inpRaw.get()),
            static_cast<const dim_t*>(devIdxStart.get()),
            static_cast<const dim_t*>(devIdxEnd.get()),
            static_cast<const dim_t*>(devOutDims.get()),
            static_cast<const dim_t*>(devIdxPtr.get()),
            static_cast<float*>(outRaw.get()));
  } else {
    throw std::invalid_argument("Index type must be one of s32/s64/u32/u64");
  }
  FL_CUDA_CHECK(cudaPeekAtLastError());

  if (outType == f16) {
    out = out.as(f16);
  }
}

} // namespace fl
