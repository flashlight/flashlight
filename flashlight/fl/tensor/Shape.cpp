/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Shape.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace fl {

Shape::Shape(std::vector<Dim> d) : dims_(std::move(d)) {}
Shape::Shape(std::initializer_list<Dim> d) : Shape(std::vector<Dim>(d)) {}

const Dim kEmptyShapeNumberOfElements = 1;

void Shape::checkDimsOrThrow(const size_t dim) const {
  if (dim > ndim() - 1) {
    std::stringstream ss;
    ss << "Shape index " << std::to_string(dim)
       << " out of bounds for shape with " << std::to_string(dims_.size())
       << " dimensions.";
    throw std::invalid_argument(ss.str());
  }
}

Dim Shape::elements() const {
  if (dims_.empty()) {
    return kEmptyShapeNumberOfElements;
  }
  return std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<Dim>());
}

int Shape::ndim() const {
  return dims_.size();
}

Dim Shape::dim(const size_t dim) const {
  checkDimsOrThrow(dim);
  return dims_[dim];
}

Dim& Shape::operator[](const size_t dim) {
  checkDimsOrThrow(dim);
  return dims_[dim];
}

const Dim& Shape::operator[](const size_t dim) const {
  checkDimsOrThrow(dim);
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

const std::vector<Dim>& Shape::get() const {
  return dims_;
}

std::vector<Dim>& Shape::get() {
  return dims_;
};

std::string Shape::toString() const {
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < ndim(); ++i) {
    ss << dim(i) << (i == ndim() - 1 ? "" : ", ");
  }
  ss << ")";
  return ss.str();
}

std::ostream& operator<<(std::ostream& ostr, const Shape& s) {
  ostr << s.toString();
  return ostr;
}

} // namespace fl
