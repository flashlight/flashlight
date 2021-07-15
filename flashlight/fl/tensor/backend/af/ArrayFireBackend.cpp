/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <af/algorithm.h>
#include <af/arith.h>
#include <af/blas.h>
#include <af/data.h>
#include <af/device.h>
#include <af/exception.h>
#include <af/gfor.h>
#include <af/lapack.h>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"

namespace {

typedef af::array (*reduceFunc_t)(const af::array&, const int);

af::array afReduceAxes(
    const af::array& input,
    const std::vector<int>& axes,
    reduceFunc_t func) {
  auto arr = input;
  for (int dim : axes) {
    arr = func(arr, dim);
  }
  return arr;
}

} // namespace

namespace fl {

ArrayFireBackend::ArrayFireBackend() {
  AF_CHECK(af_init());
}

ArrayFireBackend& ArrayFireBackend::getInstance() {
  static ArrayFireBackend instance;
  return instance;
}

/* --------------------------- Tensor Operators --------------------------- */

/************************ Shaping and Indexing *************************/
Tensor ArrayFireBackend::reshape(const Tensor& tensor, const Shape& shape) {
  return toTensor<ArrayFireTensor>(
      af::moddims(toArray(tensor), detail::flToAfDims(shape)));
}

Tensor ArrayFireBackend::transpose(
    const Tensor& tensor,
    const Shape& dims /* = {} */) {
  if (tensor.shape().nDims() == 2 &&
      (dims.nDims() == 0 || dims == Shape({1, 0}))) {
    // fastpath for matrices
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(af::transpose(toArray(tensor))));
  } else if (dims.nDims() == 0) {
    // flip all dimensions
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(af::reorder(toArray(tensor), 3, 2, 1, 0)));
  } else {
    if (dims.nDims() > AF_MAX_DIMS) {
      throw std::invalid_argument(
          "ArrayFire tensor transpose was given "
          "permutation dims with > 4 axes");
    }
    // reorder based on specified dimensions
    std::vector<dim_t> d(AF_MAX_DIMS);
    std::iota(std::begin(d), std::end(d), 0);
    for (size_t i = 0; i < dims.nDims(); ++i) {
      d[i] = dims[i];
    }
    return toTensor<ArrayFireTensor>(detail::condenseIndices(
        af::reorder(toArray(tensor), d[0], d[1], d[2], d[3])));
  }
}

Tensor ArrayFireBackend::tile(const Tensor& tensor, const Shape& shape) {
  return toTensor<ArrayFireTensor>(
      af::tile(toArray(tensor), detail::flToAfDims(shape)));
}

Tensor ArrayFireBackend::concatenate(
    const std::vector<Tensor>& tensors,
    unsigned axis) {
  if (tensors.size() > 10) {
    throw std::invalid_argument(
        "ArrayFire concatenate doesn't support > 10 tensors");
  }
  std::vector<af_array> arrs(tensors.size());
  std::transform(
      tensors.begin(), tensors.end(), arrs.begin(), [](const Tensor& t) {
        return toArray(t).get();
      });
  af_array handle = nullptr;
  AF_CHECK(af_join_many(&handle, axis, tensors.size(), arrs.data()));
  return toTensor<ArrayFireTensor>(af::array(handle));
}

Tensor ArrayFireBackend::nonzero(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::where(toArray(tensor)));
}

/************************** Unary Operators ***************************/

Tensor ArrayFireBackend::exp(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::exp(toArray(tensor)));
}

Tensor ArrayFireBackend::log(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::log(toArray(tensor)));
}

Tensor ArrayFireBackend::negative(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(-toArray(tensor));
}

Tensor ArrayFireBackend::logicalNot(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(!toArray(tensor));
}

Tensor ArrayFireBackend::log1p(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::log1p(toArray(tensor)));
}

Tensor ArrayFireBackend::sin(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::sin(toArray(tensor)));
}

Tensor ArrayFireBackend::cos(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::cos(toArray(tensor)));
}

Tensor ArrayFireBackend::sqrt(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::sqrt(toArray(tensor)));
}

Tensor ArrayFireBackend::tanh(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::tanh(toArray(tensor)));
}

Tensor ArrayFireBackend::floor(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::floor(toArray(tensor)));
}

Tensor ArrayFireBackend::ceil(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::ceil(toArray(tensor)));
}

Tensor ArrayFireBackend::absolute(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::abs(toArray(tensor)));
}

Tensor ArrayFireBackend::clip(
    const Tensor& tensor,
    const Tensor& low,
    const Tensor& high) {
  return toTensor<ArrayFireTensor>(
      af::clamp(toArray(tensor), toArray(low), toArray(high)));
}

Tensor ArrayFireBackend::isnan(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::isNaN(toArray(tensor)));
}

/************************** Binary Operators ***************************/

Tensor ArrayFireBackend::minimum(const Tensor& lhs, const Tensor& rhs) {
  return toTensor<ArrayFireTensor>(af::min(toArray(lhs), toArray(rhs)));
}

Tensor ArrayFireBackend::maximum(const Tensor& lhs, const Tensor& rhs) {
  return toTensor<ArrayFireTensor>(af::max(toArray(lhs), toArray(rhs)));
}

Tensor ArrayFireBackend::power(const Tensor& lhs, const Tensor& rhs) {
  return toTensor<ArrayFireTensor>(af::pow(toArray(lhs), toArray(rhs)));
}

/************************** Reductions ***************************/

Tensor ArrayFireBackend::amin(
    const Tensor& input,
    const std::vector<int>& axes) {
  return toTensor<ArrayFireTensor>(afReduceAxes(toArray(input), axes, af::min));
}

// TODO: consolidate with above
double ArrayFireBackend::amin(const Tensor& input) {
  return af::min<double>(toArray(input));
}

Tensor ArrayFireBackend::amax(
    const Tensor& input,
    const std::vector<int>& axes) {
  return toTensor<ArrayFireTensor>(afReduceAxes(toArray(input), axes, af::max));
}

// TODO: consolidate with above
double ArrayFireBackend::amax(const Tensor& input) {
  return af::max<double>(toArray(input));
}

Tensor ArrayFireBackend::sum(
    const Tensor& input,
    const std::vector<int>& axes) {
  return toTensor<ArrayFireTensor>(afReduceAxes(toArray(input), axes, af::sum));
}

// TODO: consolidate with above
double ArrayFireBackend::sum(const Tensor& input) {
  return af::sum<double>(toArray(input));
}

Tensor ArrayFireBackend::mean(
    const Tensor& input,
    const std::vector<int>& axes) {
  // Cannot use afReduceAxes because sum uses dim instead of int
  auto arr = toArray(input);
  for (int dim : axes) {
    arr = af::mean(arr, dim);
  }
  return toTensor<ArrayFireTensor>(std::move(arr));
}

// TODO: consolidate with above
double ArrayFireBackend::mean(const Tensor& input) {
  return af::mean<double>(toArray(input));
}

Tensor ArrayFireBackend::var(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool bias) {
  // Use arrayfire default for one dimension which may be optimized
  auto& arr = toArray(input);
  if (axes.size() == 1) {
    return toTensor<ArrayFireTensor>(af::var(
        arr, bias ? AF_VARIANCE_SAMPLE : AF_VARIANCE_POPULATION, axes[0]));
  }
  auto meanArr = mean(input, axes);
  // TODO Replace when we have batchFunc for fl::Tensor
  auto x = af::batchFunc(arr, toArray(meanArr), af::operator-);
  x = af::pow(x, 2);
  x = afReduceAxes(x, axes, af::sum);

  int denominator = 1;
  auto dims = toArray(input).dims();
  for (auto dim : axes) {
    denominator *= dims[dim];
  }
  if (bias) {
    denominator--;
  }

  x = x / denominator;
  return toTensor<ArrayFireTensor>(std::move(x));
}

// TODO: consolidate with above
double ArrayFireBackend::var(const Tensor& input, const bool bias) {
  return af::var<double>(toArray(input), bias);
}

Tensor ArrayFireBackend::std(
    const Tensor& input,
    const std::vector<int>& axes) {
  if (axes.size() == 1) {
    // Use arrayfire default for one dimension which may be optimized
    return toTensor<ArrayFireTensor>(af::stdev(toArray(input), axes[0]));
  }
  return this->sqrt(this->var(input, axes, /* bias = */ false));
}

double ArrayFireBackend::norm(const Tensor& input) {
  return af::norm(toArray(input));
}

void ArrayFireBackend::print(const Tensor& tensor) {
  af::print("", toArray(tensor));
}
} // namespace fl
