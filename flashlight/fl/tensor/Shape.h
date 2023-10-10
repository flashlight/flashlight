/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <initializer_list>
#include <limits>
#include <ostream>
#include <utility>
#include <vector>

#include "flashlight/fl/common/Defines.h"

namespace fl {

// The type of a dimension.
using Dim = long long;

/**
 * An object describing the dimensions of a tensor.
 *
 * The dimensions and sizes of a shape are explicit; where some tensor libraries
 * implement implicit dimensions (i.e. those that are 1 are ignored), Flashlight
 * Shapes can be of arbitrary size and 1-dimensions distinguish them.
 * Concretely, (3, 1) and (3) are distinct shapes. See ShapeTest for further
 * examples.
 *
 * Shapes dimensions should be >= 1 in size. Shapes with a zero dimension have
 * zero elements, even if other dimensions are of nonzero size. For example: a
 * Shape of (0) has zero elements, as does a Shape with dimensions (1, 2, 3, 0).
 *
 * Different tensor backends implement different shape and dimension semantics.
 * As such, these need to be converted back and forth to and from Flashlight
 * Shapes. Having a common set of behaviors in this API ensures that tensors and
 * their shapes can be freely-manipulated across tensor backends.
 *
 * Shape is an interface and can be derived from or implemented given specific
 * backing storage or handles.
 */
class FL_API Shape {
  // Storage for the dimension values. Defaults to an empty Shape {0}, whereas
  // {} is a scalar shape.
  std::vector<Dim> dims_;

  /**
   * Check if a dimension is valid (i.e. in bounds) given the current size of
   * the shape. If not valid, throws an exception.
   */
  void checkDimsOrThrow(const size_t dim) const;

 public:
  Shape() = default;
  ~Shape() = default;
  /**
   * Gives the maximum number of dimensions a tensor of a particular shape can
   * have.
   *
   * If the maximum size can be arbitrarily high, `std::numeric_limits<Dim>`
   * should be used.
   */
  static constexpr size_t kMaxDims = std::numeric_limits<size_t>::max();

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
  Dim elements() const;

  /**
   * @return Number of dimensions in the shape.
   */
  int ndim() const;

  /**
   * Get the size of a given dimension in the number of arguments. Throws if the
   * given dimension is larger than the number of dimensions.
   *
   * @return the number of elements at the given dimension
   */
  Dim dim(const size_t dim) const;

  /**
   * Returns a reference to the given index
   */
  Dim& operator[](const size_t dim);
  const Dim& operator[](const size_t dim) const;

  /**
   * Compares two shapes. Returns true if their dim vectors are equal.
   */
  bool operator==(const Shape& other) const;
  bool operator!=(const Shape& other) const;

  /**
   * Compare a shape to an initializer list.
   */
  bool operator==(const std::initializer_list<Dim>& other) const;
  bool operator!=(const std::initializer_list<Dim>& other) const;

  /**
   * Gets a reference to the underying dims vector.
   */
  const std::vector<Dim>& get() const;
  std::vector<Dim>& get();

  /**
   * Returns a string representation of the Shape
   */
  std::string toString() const;
};

/**
 * Write a shape representation to an output stream.
 */
FL_API std::ostream& operator<<(std::ostream& ostr, const Shape& s);

} // namespace fl
