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
  Tensor amax(const Tensor& input, const std::vector<int>& axes) override;
  Tensor sum(const Tensor& input, const std::vector<int>& axes) override;
  Tensor mean(const Tensor& input, const std::vector<int>& axes) override;
  Tensor var(const Tensor& input, const std::vector<int>& axes, bool bias)
      override;
  double norm(const Tensor& input) override;
};

} // namespace fl
