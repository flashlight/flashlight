/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <af/algorithm.h>
#include <af/array.h>
#include <af/statistics.h>

#include "flashlight/fl/tensor/TensorAdapter.h"

namespace fl {

/**
 * Tensor adapter for the ArrayFire tensor library. Maps operations expressed in
 * Flashlight Tensors to ArrayFire
 */
class ArrayFireTensor : public TensorAdapterBase {
  // The internal ArrayFire array handle
  af::array array_;

 public:
  /**
   * Constructs an ArrayFireTensor.
   *
   * Since af::arrays are refcounted, an instance of this class
   * can only be created using arrays that are moved therein.
   *
   * Tensor operations occurring directly on this tensor's underlying af::array
   * should not copy the array else take a performance penalty (via an internal
   * copy if refcount is > 1 in some cases).
   *
   * @param[in] array&& construct a tensor from an ArrayFire array rvalue
   * reference.
   */
  explicit ArrayFireTensor(af::array&& array);
  ~ArrayFireTensor() override = default;

  TensorBackend backend() const override {
    return TensorBackend::ArrayFire;
  }

  Shape shape() const override;
  dtype type() const override;
  Tensor astype(const dtype type) override;

  af::array& getHandle() {
    return array_;
  }

  const af::array& getHandle() const {
    return array_;
  }
};

/**
 * Gets an af::array from a Tensor. If the Tensor is not ArrayFire-backed,
 * throws an exception
 *
 * @param[in] tensor the input tensor
 * @return the array underying the Tensor
 */
af::array& toArray(const Tensor& tensor);

/************************** Generic Reductions ***************************/
/*
 * TODO: move me to a singleton backend abstraction interface once available so
 * generics can be properly handled.
 */

/**
 * Compute the minimum value across all axes.
 *
 * @param[in] input the input along which to operate
 * @return a scalar T containing the mim
 *
 */
template <typename T>
T amin(const Tensor& input) {
  return af::min<T>(toArray(input));
}

/**
 * Compute the maximum value across all axes.
 *
 * @param[in] input the input along which to operate
 * @return a scalar T containing the max value
 *
 */
template <typename T>
T amax(const Tensor& input) {
  return af::max<T>(toArray(input));
}

/**
 * Sum of array over all axes.
 *
 * @param[in] input the input along which to operate
 * @return a scalar T containing the sum
 */
template <typename T>
T sum(const Tensor& input) {
  return af::sum<T>(toArray(input));
}

/**
 * Mean of array over all axes.
 *
 * @param[in] input the input along which to operate
 * @return a scalar T containing the mean
 */
template <typename T>
T mean(const Tensor& input) {
  return af::mean<T>(toArray(input));
}

/**
 * var of array over all axes.
 *
 * @param[in] input the input along which to operate
 * @return a scalar T containing the var
 */
template <typename T>
T var(const Tensor& input, bool bias = false) {
  return af::var<T>(toArray(input), bias);
}

} // namespace fl
