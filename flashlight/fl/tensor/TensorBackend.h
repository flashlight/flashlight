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
 * A Tensor backend that can be used to store global state associated with a
 * particular tensor implementation.
 *
 * This abstraction facilitates adherence to the implementation requirements for
 * global operators that operate on tensors (e.g. those functions that are not
 * members of `fl::Tensor`).
 *
 * Flashlight Tensors dispatch to their corresponding backends using
 * fl::Tensor::backend() --> typeToBackend (see below) to grab the correct
 * singleton.
 */
class TensorBackend {
 public:
  TensorBackend() = default;
  virtual ~TensorBackend() = default;

  /* --------------------------- Tensor Operators ---------------------------
   * For operator documentation and expected behavior, see TensorBase.h.
   */
  /************************ Shaping and Indexing *************************/
  virtual Tensor reshape(const Tensor& tensor, const Shape& shape) = 0;
  virtual Tensor transpose(
      const Tensor& tensor,
      const Shape& dims /* = {} */) = 0;
  virtual Tensor tile(const Tensor& tensor, const Shape& shape) = 0;
  virtual Tensor concatenate(const std::vector<Tensor>& tensors, unsigned axis) = 0;

  /************************** Unary Operators ***************************/
  virtual Tensor exp(const Tensor& tensor) = 0;
  virtual Tensor log(const Tensor& tensor) = 0;
  virtual Tensor negative(const Tensor& tensor) = 0;
  virtual Tensor logicalNot(const Tensor& tensor) = 0;
  virtual Tensor log1p(const Tensor& tensor) = 0;
  virtual Tensor sin(const Tensor& tensor) = 0;
  virtual Tensor cos(const Tensor& tensor) = 0;
  virtual Tensor sqrt(const Tensor& tensor) = 0;
  virtual Tensor tanh(const Tensor& tensor) = 0;
  virtual Tensor absolute(const Tensor& tensor) = 0;
  virtual Tensor
  clip(const Tensor& tensor, const Tensor& low, const Tensor& high) = 0;
  virtual Tensor isnan(const Tensor& tensor) = 0;

  /************************** Binary Operators ***************************/
  virtual Tensor minimum(const Tensor& lhs, const Tensor& rhs) = 0;
  virtual Tensor maximum(const Tensor& lhs, const Tensor& rhs) = 0;
  virtual Tensor power(const Tensor& lhs, const Tensor& rhs) = 0;

  /************************** Reductions ***************************/
  virtual Tensor amin(const Tensor& input, const std::vector<int>& axes) = 0;
  virtual double amin(const Tensor& input) = 0; // TODO: consoildate w/ above
  virtual Tensor amax(const Tensor& input, const std::vector<int>& axes) = 0;
  virtual double amax(const Tensor& input) = 0; // TODO: consoildate w/ above
  virtual Tensor sum(const Tensor& input, const std::vector<int>& axes) = 0;
  virtual double sum(const Tensor& input) = 0; // TODO: consolidate w/ above
  virtual Tensor mean(const Tensor& input, const std::vector<int>& axes) = 0;
  virtual double mean(const Tensor& input) = 0; // TODO: consolidate w/ above
  virtual Tensor
  var(const Tensor& input, const std::vector<int>& axes, bool bias) = 0;
  virtual double var(
      const Tensor& input,
      bool bias) = 0; // TODO: consolidate w/ above
  virtual double norm(const Tensor& input) = 0;

  /************************** Utils ***************************/
  virtual void print(const Tensor& tensor) = 0;
};

namespace detail {

/**
 * Compare the backends of two tensors.
 *
 * @return true if the backends of both tensors are the same, else false.
 */
bool areBackendsEqual(const Tensor& a, const Tensor& b);

/**
 * Compare the backends of multiple tensors.
 *
 * @return true if all tensors' backends are the same, false otherwise.
 */
template <typename... Args>
bool areBackendsEqual(const Tensor& a, const Tensor& b, const Args&... args) {
  return areBackendsEqual(a, b) && areBackendsEqual(a, args...) &&
      areBackendsEqual(b, args...);
}

} // namespace detail
} // namespace fl
