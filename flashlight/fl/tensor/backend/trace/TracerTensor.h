/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <ostream>
#include <stdexcept>

// #include "flashlight/fl/tensor/TensorBackendTraits.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/trace/DefaultTracer.h"
#include "flashlight/fl/tensor/backend/trace/TracerBackend.h"
#include "flashlight/fl/tensor/backend/trace/TracerBackendBase.h"
#include "flashlight/fl/tensor/backend/trace/TracerTensorBase.h"

namespace fl {

/**
 * A backend that wraps another backend and provides tracing info on top of it.
 *
 * TODO: parameterize this only in terms of one generic!
 */
template <typename T, typename U>
class TracerTensor : public TracerTensorBase {
  std::unique_ptr<Tensor> tensor_;

 public:
  /**
   * Construct a TracerTensor using some data.
   *
   * @param[in] shape the shape of the new tensor
   * @param[in] ptr the buffer containing underlying tensor data
   * @param[in] type the type of the new tensor
   * @param[in] memoryLocation the location of the buffer
   */
  TracerTensor(
      const Shape& shape,
      fl::dtype type,
      const void* ptr,
      Location memoryLocation) {
    tensor_ = std::make_unique<Tensor>(
        std::make_unique<T>(shape, type, ptr, memoryLocation));
  }

  // Constructor for a sparse TracerTensor. Can throw if unimplemented.
  TracerTensor(
      const Dim nRows,
      const Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType) {
    tensor_ = std::make_unique<Tensor>(
        std::make_unique<T>(nRows, nCols, values, rowIdx, colIdx, storageType));
  }

  Tensor& tensor() {
    return *tensor_;
  }

  TensorBackend& tracedBackend() const override {
    // TODO: add a static_assert here that this method exists via sfinae
    return toTensor<T>().backend();
  }

  virtual TracerBackendBase& backend() const override {
    return U::getInstance();
    // return TracerBackend<fl::tensor_traits<T>::backend_type>::getInstance();
  }
};

} // namespace fl
