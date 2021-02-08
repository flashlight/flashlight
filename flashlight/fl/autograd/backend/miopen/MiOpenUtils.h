/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>
#include <miopen/miopen.h>

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/Defines.h"

namespace fl {

miopenDataType_t miopenMapToType(const af::dtype& t);

class TensorDescriptor {
 public:
  explicit TensorDescriptor(const af::array& a);
  explicit TensorDescriptor(const Variable& a);

  TensorDescriptor(const af::dtype type, const af::dim4& af_dims);

  miopenTensorDescriptor_t descriptor;
  ~TensorDescriptor();
};

class TensorDescriptorArray {
 public:
  TensorDescriptorArray(int size, const af::dtype type, const af::dim4& dims);

  miopenTensorDescriptor_t* descriptors;
  ~TensorDescriptorArray();

 private:
  std::vector<TensorDescriptor> desc_vec;
  std::vector<miopenTensorDescriptor_t> desc_raw_vec;
};

class FilterDescriptor {
 public:
  explicit FilterDescriptor(const af::array& a);
  explicit FilterDescriptor(const Variable& a);
  miopenTensorDescriptor_t descriptor;
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
  miopenConvolutionDescriptor_t descriptor;
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
  miopenPoolingDescriptor_t descriptor;
  ~PoolingDescriptor();
};

class DropoutDescriptor {
 public:
  explicit DropoutDescriptor(float drop_prob);
  miopenDropoutDescriptor_t descriptor;
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
  miopenRNNDescriptor_t descriptor;
  ~RNNDescriptor();
};

const void* kOne(const af::dtype t);
const void* kZero(const af::dtype t);
miopenHandle_t getMiOpenHandle();

constexpr size_t kWorkspaceSizeLimitBytes = 512 * 1024 * 1024; // 512 MB

} // namespace fl
