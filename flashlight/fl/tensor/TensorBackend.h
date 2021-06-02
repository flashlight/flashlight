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
 * members of `fl::Tensor`). The interface defines here implicitly defines the
 * required functionality.
 *
 * Flashlight Tensors dispatch to their corresponding backends using
 * typeToBackend (see below) to grab the correct singleton.
 */
class TensorBackend {
 public:
  TensorBackend() = default;
  virtual ~TensorBackend() = default;

  /* --------------------------- Tensor Operators ---------------------------
   * For operator documentation and expected behavior, see TensorBase.h.
   */

  /************************** Unary Operators ***************************/
  virtual Tensor exp(const Tensor& tensor) = 0;
  virtual Tensor log(const Tensor& tensor) = 0;
};

} // namespace fl
