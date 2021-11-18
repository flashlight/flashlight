/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Shape.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace fl {

Shape::Shape(std::vector<Dim> d) : dims_(std::move(d)) {}
Shape::Shape(std::initializer_list<Dim> d) : Shape(std::vector<Dim>(d)) {}

const Dim kEmptyShapeNumberOfElements = 1;

Dim Shape::elements() const {
  if (dims_.size() == 0) {
    return kEmptyShapeNumberOfElements;
  }
  return std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<Dim>());
}

size_t Shape::ndim() const {
  return dims_.size();
}

Dim Shape::dim(const size_t dim) const {
  if (dim >= dims_.size()) {
    throw std::invalid_argument(
        "fl::Shape::dim - passed dimension is larger than "
        "the number of dimensions in the shape");
  }
  return dims_[dim];
}

Dim& Shape::operator[](const size_t dim) {
  return dims_[dim];
}

const Dim& Shape::operator[](const size_t dim) const {
  return dims_[dim];
}

bool Shape::operator==(const Shape& other) const {
  return dims_ == other.dims_;
}

bool Shape::operator!=(const Shape& other) const {
  return !(this->operator==(other));
}

bool Shape::operator==(const std::initializer_list<Dim>& other) const {
  return dims_.size() == other.size() &&
      std::equal(std::begin(dims_), std::end(dims_), std::begin(other));
}

bool Shape::operator!=(const std::initializer_list<Dim>& other) const {
  return !(this->operator==(other));
}

std::ostream& operator<<(std::ostream& ostr, const Shape& s) {
  ostr << "(";
  for (size_t i = 0; i < s.ndim(); ++i) {
    ostr << s.dim(i) << (i == s.ndim() - 1 ? "" : ", ");
  }
  ostr << ")";
  return ostr;
}

const std::vector<Dim>& Shape::get() const {
  return dims_;
}

std::vector<Dim>& Shape::get() {
  return dims_;
};

} // namespace fl
