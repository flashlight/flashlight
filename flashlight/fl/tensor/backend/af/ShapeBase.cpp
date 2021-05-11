/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/ShapeBase.h"

#include <stdexcept>

#include <af/dim4.hpp>

namespace fl {

Shape::Shape(std::vector<Dim> d) {
  if (d.size() > 4) {
    throw std::invalid_argument(
        "Shape::Shape - cannot construct an ArrayFire shape"
        " with more than four dimensions.");
  }
  std::vector<dim_t> dims(4, 1);
  for (size_t i = 0; i < d.size(); ++i) {
    dims[i] = d[i];
  }
  dims_ = af::dim4(dims[0], dims[1], dims[2], dims[3]);
}

Shape::Shape(std::initializer_list<Dim> d) : Shape(std::vector<dim_t>(d)) {}

size_t Shape::elements() const {
  return dims_.elements();
}

size_t Shape::nDims() const {
  return dims_.ndims();
}

Dim Shape::dim(size_t dim) const {
  if (dim > 3) {
    throw std::invalid_argument(
        "fl::Shape::dim - passed dimension is larger than "
        "the number of dimensions in the shape");
  }
  return dims_[dim];
}

bool Shape::operator==(const Shape& other) const {
  return dims_ == other.dims_;
}

bool Shape::operator!=(const Shape& other) const {
  return !(this->operator==(other));
}

std::ostream& operator<<(std::ostream& ostr, const Shape& s) {
  ostr << s.dims_;
  return ostr;
}

} // namespace fl
