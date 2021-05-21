/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

/**
 * Convert a Tensor from an adapter or via arguments for an adapter.
 *
 * @param[in] t the thing to convert
 * @return a Tensor containing the ArrayFire array
 */
template <typename Impl, typename... T>
Tensor toTensor(T&&... t) {
  return Tensor(std::make_unique<Impl>(std::forward<T>(t)...));
}

/**
 * The implementation interface for Flashlight Tensor backends.
 *
 * Defines the implementation requirements and behaviors for a particular tensor
 * backend. These implementations can be tested against default
 * backend/reference implementations.
 *
 * Derived types from TensorAdapterBase should not be constructed or operated on
 * literally. Calls to their member implementations are dispatched via visiting
 * Tensor or other interfaces.
 */
class TensorAdapterBase {
 public:
  TensorAdapterBase() = default;
  virtual ~TensorAdapterBase() = default;

  /**
   * Gets the tensor's associated backend.
   *
   * @return TensorBackend enum associated with the backend
   */
  virtual TensorBackend backend() const = 0;

  /**
   * Get the shape of a tensor.
   *
   * @return the shape of the tensor
   */
  virtual Shape shape() const = 0;

  /**
   * Get the data type of tensor.
   *
   * @return the dtype of the tensor
   */
  virtual dtype type() const = 0;

  /**
   * Returns a tensor with elements cast as a particular type
   *
   * @param[in] the type to which to cast the tensor
   * @return a tensor with element-wise cast to the new type
   */
  virtual Tensor astype(const dtype type) = 0;
};

} // namespace fl
