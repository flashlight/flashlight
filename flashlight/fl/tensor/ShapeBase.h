/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ostream>
#include <utility>
#include <vector>

// TODO:fl::Tensor {af} remove me when ArrayFire is a particular subimpl
#include <af/dim4.hpp>

namespace fl {

// The type of a dimension.
using Dim = long long;

/**
 * A base interface for defining some notion of shape on top of a tensor. Can be
 * constructed from an initializer list, inspected, and modified.
 *
 * TODO: for now, Shape has a af::dim4 member that hides core logic given that
 * the default tensor lib implementation is ArrayFire. This will be removed once
 * the transition to fl::Tensor across Flashlight is complete.
 *
 * TODO: make these methods pure virtual when migration is complete.
 */
struct Shape {
  virtual ~Shape() = default;
  Shape() = default;

  // TODO:fl::Tensor {af} remove these when ArrayFire is a particular subimpl
  af::dim4 dims_;
  /**
   * Gives the maximum number of dimensions a tensor of a particular shape can
   * have.
   *
   * If the maximum size can be arbitrarily high, `std::numeric_limits<Dim>`
   * should be used.
   *
   * TODO: remove me once migration is complete.
   */
  static constexpr size_t kMaxDims{AF_MAX_DIMS};

  /**
   * Initialize a Shape via a vector.
   */
  explicit Shape(std::vector<Dim> d);

  /**
   * Initialize a Shape via an initializer list.
   */
  /* implicit */ Shape(std::initializer_list<Dim> d);

  /**
   * @return the number of elements in a tensor that has the given shape.
   */
  virtual size_t elements() const;

  /**
   * @return Number of dimensions in the shape.
   */
  virtual size_t nDims() const;

  /**
   * Get the size of a given dimension. Throws if the given dimension is larger
   * than the number of dimensions.
   *
   * @return the Dim at the given dimension
   */
  virtual Dim dim(size_t dim) const;

  /**
   * Compares two shapes. Returns true if their dim vectors are equal.
   */
  virtual bool operator==(const Shape& other) const;
  virtual bool operator!=(const Shape& other) const;
};

std::ostream& operator<<(std::ostream& ostr, const Shape& s);

} // namespace fl
