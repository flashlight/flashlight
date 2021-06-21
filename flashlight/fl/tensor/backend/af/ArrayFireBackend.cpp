/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <af/algorithm.h>
#include <af/arith.h>
#include <af/data.h>
#include <af/device.h>
#include <af/exception.h>
#include <af/gfor.h>
#include <af/lapack.h>

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"

/*
 * TODO: this is duplicative - remove this from flashlight/fl/common/Utils.h
 * once the rest of the proj depends on headers here.
 */
#define AF_CHECK(fn)                                                          \
  do {                                                                        \
    af_err __err = fn;                                                        \
    if (__err == AF_SUCCESS) {                                                \
      break;                                                                  \
    }                                                                         \
    throw af::exception(                                                      \
        "ArrayFire error: ", __PRETTY_FUNCTION__, __FILE__, __LINE__, __err); \
  } while (0)

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
    bool bias) {
  // Use arrayfire default for one dimension which may be optimized
  auto& arr = toArray(input);
  if (axes.size() == 1) {
    return toTensor<ArrayFireTensor>(af::var(arr, bias, axes[0]));
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
double ArrayFireBackend::var(const Tensor& input, bool bias) {
  return af::var<double>(toArray(input), bias);
}

double ArrayFireBackend::norm(const Tensor& input) {
  return af::norm(toArray(input));
}

void ArrayFireBackend::print(const Tensor& tensor) {
  af::print("", toArray(tensor));
}
} // namespace fl
