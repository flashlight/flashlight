/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <mutex>
#include <stdexcept>

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/trace/DefaultTracer.h"
#include "flashlight/fl/tensor/backend/trace/TracerBackend.h"
#include "flashlight/fl/tensor/backend/trace/TracerBackendBase.h"
#include "flashlight/fl/tensor/backend/trace/TracerTensorBase.h"

namespace fl {

/**
 * A backend that wraps another backend and provides tracing info on top of it.
 */
template <typename T, typename U>
class TracerTensor : public TracerTensorBase {
  static inline std::once_flag initTensorCreator_;

  TracerBackendBase::TensorCreatorFunc getTensorCreator() {
    return [](Tensor&& t) -> Tensor {
      return Tensor(std::make_unique<TracerTensor<T, U>>(std::move(t)));
    };
  }

 public:
  TracerTensor() : TracerTensorBase(std::make_unique<T>()) {
    backend().setTensorCreator(getTensorCreator());
  }

  TracerTensor(Tensor&& t) : TracerTensorBase(std::move(t)) {
    backend().setTensorCreator(getTensorCreator());
  }

  TracerTensor(
      const Shape& shape,
      fl::dtype type,
      const void* ptr,
      Location memoryLocation)
      : TracerTensorBase(
            Tensor(std::make_unique<T>(shape, type, ptr, memoryLocation))) {
    backend().setTensorCreator(getTensorCreator());
  }

  TracerTensor(
      const Dim nRows,
      const Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType)
      : TracerTensorBase(
            Tensor(std::make_unique<
                   T>(nRows, nCols, values, rowIdx, colIdx, storageType))) {
    backend().setTensorCreator(getTensorCreator());
  }

  TensorBackend& tracedBackend() const override {
    return toTensor<T>().backend();
  }

  virtual TracerBackendBase& backend() const override {
    return TracerBackend<T, U>::getInstance();
  }
};

} // namespace fl
