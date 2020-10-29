/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>
#include <cudnn.h>

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/Defines.h"

namespace fl {

class TensorDescriptor {
 public:
  explicit TensorDescriptor(const af::array& a);
  explicit TensorDescriptor(const Variable& a);

  TensorDescriptor(const af::dtype type, const af::dim4& af_dims);

  cudnnTensorDescriptor_t descriptor;
  ~TensorDescriptor();
};

class TensorDescriptorArray {
 public:
  TensorDescriptorArray(int size, const af::dtype type, const af::dim4& dims);

  cudnnTensorDescriptor_t* descriptors;
  ~TensorDescriptorArray();

 private:
  std::vector<TensorDescriptor> desc_vec;
  std::vector<cudnnTensorDescriptor_t> desc_raw_vec;
};

class FilterDescriptor {
 public:
  explicit FilterDescriptor(const af::array& a);
  explicit FilterDescriptor(const Variable& a);
  cudnnFilterDescriptor_t descriptor;
  ~FilterDescriptor();
};

class ConvDescriptor {
 public:
  ConvDescriptor(
      af::dtype type,
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

  af::array& getDropoutStates();
};

class RNNDescriptor {
 public:
  RNNDescriptor(
      af::dtype type,
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

cudnnDataType_t cudnnMapToType(const af::dtype& t);

const void* kOne(const af::dtype t);

const void* kZero(const af::dtype t);

cudnnHandle_t getCudnnHandle();

} // namespace fl
