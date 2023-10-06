/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cudnn.h>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

class TensorDescriptor {
 public:
  explicit TensorDescriptor(const Tensor& a);

  TensorDescriptor(const fl::dtype type, const Shape& af_dims);

  cudnnTensorDescriptor_t descriptor;
  ~TensorDescriptor();
};

class TensorDescriptorArray {
 public:
  TensorDescriptorArray(int size, const fl::dtype type, const Shape& dims);

  cudnnTensorDescriptor_t* descriptors;
  ~TensorDescriptorArray();

 private:
  std::vector<TensorDescriptor> desc_vec;
  std::vector<cudnnTensorDescriptor_t> desc_raw_vec;
};

class FilterDescriptor {
 public:
  explicit FilterDescriptor(const Tensor& a);
  cudnnFilterDescriptor_t descriptor;
  ~FilterDescriptor();
};

class ConvDescriptor {
 public:
  ConvDescriptor(
      fl::dtype type,
      int px,
      int py,
      int sx,
      int sy,
      int dx,
      int dy,
      int groups = 1);
  cudnnConvolutionDescriptor_t descriptor;
  ~ConvDescriptor();
};

class PoolingDescriptor {
 public:
  PoolingDescriptor(
      int wx,
      int wy,
      int sx,
      int sy,
      int px,
      int py,
      PoolingMode mode);
  cudnnPoolingDescriptor_t descriptor;
  ~PoolingDescriptor();
};

class DropoutDescriptor {
 public:
  explicit DropoutDescriptor(float drop_prob);
  cudnnDropoutDescriptor_t descriptor;
  ~DropoutDescriptor();

  Tensor& getDropoutStates();
};

class RNNDescriptor {
 public:
  RNNDescriptor(
      fl::dtype type,
      int hidden_size,
      int num_layers,
      RnnMode mode,
      bool bidirectional,
      DropoutDescriptor& dropout);
  cudnnRNNDescriptor_t descriptor;
  ~RNNDescriptor();
};

#define CUDNN_CHECK_ERR(expr) ::fl::cudnnCheckErr((expr))

void cudnnCheckErr(cudnnStatus_t status);

cudnnDataType_t cudnnMapToType(const fl::dtype& t);

const void* kOne(const fl::dtype t);

const void* kZero(const fl::dtype t);

// TODO: move this to CudnnAutogradExtension if we make it a singleton
cudnnHandle_t getCudnnHandle();
const CUDAStream& getCudnnStream();

} // namespace fl
