/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <af/array.h>

namespace fl {

/**
 * A base tensor interface under which implementations can define tensor
 * operations. These operations may differ across tensor libraries,
 * hardware-specific libraries or compilers and DSLs.
 *
 * TODO:fl::Tensor {doc} more documentation. For now, this class serves as a
 * simple shim between tensor operations expressed in Flashlight and underlying
 * tensor computations in ArrayFire; for now, this API will wrap the ArrayFire
 * API using semantics similar to
 * [numpy](https://numpy.org/doc/stable/reference/) and translating those into
 * ArrayFire semantics. NOTE: this API will break frequently and is not yet
 * stable.
 */
class Tensor {
  // TODO:fl::Tensor {af} remove me when ArrayFire is a particular subimpl
  af::array array_;

 public:
  /**
   * Temporary. Since af::arrays are refcounted, an instance of this class
   * should only be created using arrays that are moved therein. Tensor
   * operations occurring on that array, while adapting functions in Flashlight,
   * should operate on references and should not copy the array else take a
   * performance penalty (via an internal copy if refcount is > 1 in some
   * cases).
   *
   * @param[in] array&& construct a tensor from an ArrayFire array rvalue
   * reference.
   */
  explicit Tensor(af::array&& array);

  // TODO:fl::Tensor {af} remove me when not dependent on AF
  af::array& getArray();
};

} // namespace fl
