/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorBackend.h"

namespace fl {

/**
 * A tensor backend implementation of the ArrayFire tensor library.
 *
 * Given that ArrayFire has an internal DeviceManager singleton to manage its
 * global state, nothing is stored here as those internals are opaquely handled.
 * This class simply dispatches operations on global tensor functions to their
 * ArrayFire counterparts.
 */
class ArrayFireBackend : public TensorBackend {
  // TODO: consolidate the ArrayFire memory manager here so its global state can
  // be stored/we can reduce the number of singletons.
 public:
  ArrayFireBackend();
  ~ArrayFireBackend() override = default;

  static ArrayFireBackend& getInstance();

  /* --------------------------- Tensor Operators --------------------------- */

  /************************ Shaping and Indexing *************************/
  Tensor reshape(const Tensor& tensor, const Shape& shape) override;
  Tensor transpose(const Tensor& tensor, const Shape& dims /* = {} */) override;
  Tensor tile(const Tensor& tensor, const Shape& shape) override;
  Tensor concatenate(const std::vector<Tensor>& tensors, unsigned axis)
      override;

  /************************** Unary Operators ***************************/
  Tensor exp(const Tensor& tensor) override;
  Tensor log(const Tensor& tensor) override;
  Tensor negative(const Tensor& tensor) override;
  Tensor logicalNot(const Tensor& tensor) override;
  Tensor log1p(const Tensor& tensor) override;
  Tensor sin(const Tensor& tensor) override;
  Tensor cos(const Tensor& tensor) override;
  Tensor sqrt(const Tensor& tensor) override;
  Tensor tanh(const Tensor& tensor) override;
  Tensor absolute(const Tensor& tensor) override;
  Tensor clip(const Tensor& tensor, const Tensor& low, const Tensor& high)
      override;
  Tensor isnan(const Tensor& tensor) override;

  /************************** Binary Operators ***************************/
  Tensor minimum(const Tensor& lhs, const Tensor& rhs) override;
  Tensor maximum(const Tensor& lhs, const Tensor& rhs) override;
  Tensor power(const Tensor& lhs, const Tensor& rhs) override;

  /************************** Reductions ***************************/
  Tensor amin(const Tensor& input, const std::vector<int>& axes) override;
  double amin(const Tensor& input) override; // TODO: consolidate w/ above
  Tensor amax(const Tensor& input, const std::vector<int>& axes) override;
  double amax(const Tensor& input) override; // TODO: consolidate w/ above
  Tensor sum(const Tensor& input, const std::vector<int>& axes) override;
  double sum(const Tensor& input) override; // TODO: consolidate w/ above
  Tensor mean(const Tensor& input, const std::vector<int>& axes) override;
  double mean(const Tensor& input) override; // TODO: consolidate w/ above
  Tensor var(const Tensor& input, const std::vector<int>& axes, const bool bias)
      override;
  double var(const Tensor& input, const bool bias)
      override; // TODO: consolidate w/ above
  Tensor std(const Tensor& input, const std::vector<int>& axes, const bool bias)
      override;
  double norm(const Tensor& input) override;

  /************************** Utils ***************************/
  void print(const Tensor& tensor) override;
};

} // namespace fl
