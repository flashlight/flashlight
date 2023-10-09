/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <algorithm>

#include <af/arith.h>
#include <af/gfor.h>

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"

namespace fl {
namespace {

using reduceFunc_t = af::array (*)(const af::array&, const int);

template <typename T = reduceFunc_t>
af::array afReduceAxes(
    const af::array& input,
    const std::vector<int>& axes,
    T func,
    const bool keepDims = false) {
  auto arr = input;
  for (int dim : axes) {
    arr = func(arr, dim);
  }
  return fl::detail::condenseIndices(arr, keepDims);
}

unsigned
getReducedNumDims(unsigned inSize, unsigned axisSize, const bool keepDims) {
  if (keepDims) {
    return inSize;
  } else {
    if (inSize < axisSize) {
      return 0;
    } else {
      return inSize - axisSize;
    }
  }
}

bool isAllAxisReduction(const Tensor& input, const std::vector<int>& axes) {
  if (input.ndim() == 0 || axes.empty()) {
    return true;
  }
  if (input.ndim() != axes.size()) {
    return false;
  }
  // Check that all dims are present
  auto _axes = axes;
  std::sort(_axes.begin(), _axes.end());
  for (size_t i = 0; i < _axes.size(); ++i) {
    if (_axes[i] != i) {
      return false;
    }
  }
  return true;
}
} // namespace

Tensor ArrayFireBackend::amin(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::min<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::min(af::min(af::min(af::min(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    return toTensor<ArrayFireTensor>(
        afReduceAxes(toArray(input), axes, af::min, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::amax(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::max<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::max(af::max(af::max(af::max(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    return toTensor<ArrayFireTensor>(
        afReduceAxes(toArray(input), axes, af::max, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

void ArrayFireBackend::min(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  af::min(toArray(values), toArray(indices), toArray(input), axis);
  values = toTensor<ArrayFireTensor>(
      detail::condenseIndices(toArray(values), keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
  indices = toTensor<ArrayFireTensor>(
      detail::condenseIndices(toArray(indices), keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
}

void ArrayFireBackend::max(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  af::max(toArray(values), toArray(indices), toArray(input), axis);
  values = toTensor<ArrayFireTensor>(
      detail::condenseIndices(toArray(values), keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
  indices = toTensor<ArrayFireTensor>(
      detail::condenseIndices(toArray(indices), keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
}

Tensor ArrayFireBackend::sum(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::sum<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::sum(af::sum(af::sum(af::sum(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    return toTensor<ArrayFireTensor>(
        afReduceAxes(toArray(input), axes, af::sum, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::cumsum(const Tensor& input, const unsigned axis) {
  return toTensor<ArrayFireTensor>(
      af::accum(toArray(input), axis), /* numDims = */ input.ndim());
}

Tensor ArrayFireBackend::argmax(
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  af::array tmpVal, indices;
  af::max(tmpVal, indices, toArray(input), axis);
  return toTensor<ArrayFireTensor>(
      detail::condenseIndices(indices, keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
}

Tensor ArrayFireBackend::argmin(
    const Tensor& input,
    const unsigned axis,
    const bool keepDims) {
  af::array tmpVal, indices;
  af::min(tmpVal, indices, toArray(input), axis);
  return toTensor<ArrayFireTensor>(
      detail::condenseIndices(indices, keepDims),
      getReducedNumDims(input.ndim(), 1, keepDims));
}

Tensor ArrayFireBackend::mean(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::mean<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::mean(af::mean(af::mean(af::mean(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    return toTensor<ArrayFireTensor>(
        afReduceAxes<af::array(const af::array&, const dim_t)>(
            toArray(input), axes, af::mean, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::median(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::median<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    double median = af::median<double>(toArray(input));
    return toTensor<ArrayFireTensor>(
        af::constant(median, 1),
        /* numDims = */ 0);
  } else {
    return toTensor<ArrayFireTensor>(
        afReduceAxes<af::array(const af::array&, const dim_t)>(
            toArray(input), axes, af::median, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::var(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool bias,
    const bool keepDims) {
  af_var_bias biasMode = bias ? AF_VARIANCE_SAMPLE : AF_VARIANCE_POPULATION;
  // Use ArrayFire default for one dimension which may be optimized
  auto& arr = toArray(input);
  // Reduce along all axes returning a singleton tensor
  // TODO: modify this to af::var<af::array> to take advantage of the
  // ArrayFire reduce_all kernels once available
  if (isAllAxisReduction(input, axes)) {
    double out = af::var<double>(toArray(input), biasMode);
    return toTensor<ArrayFireTensor>(af::constant(out, 1), /* numDims = */ 0);
  } else if (axes.size() == 1) {
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(af::var(arr, biasMode, axes[0]), keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  } else {
    auto meanArr = mean(input, axes, /* keepDims = */ true);
    auto x = af::batchFunc(arr, toArray(meanArr), af::operator-);

    x = af::pow(x, 2);
    x = afReduceAxes(x, axes, af::sum, /* keepDims = */ true);

    int denominator = 1;
    auto dims = arr.dims();
    for (auto dim : axes) {
      denominator *= dims[dim];
    }
    if (bias) {
      denominator--;
    }

    x = x / denominator;
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(x, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::std(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  const bool bias = false; // TODO: make this configurable
  af_var_bias biasMode = bias ? AF_VARIANCE_SAMPLE : AF_VARIANCE_POPULATION;
  if (isAllAxisReduction(input, axes)) {
    // TODO: update to af::stdev<af::array> once specialization is available
    double out = af::stdev<double>(toArray(input), biasMode);
    return toTensor<ArrayFireTensor>(af::constant(out, 1), /* numDims = */ 0);
  } else if (axes.size() == 1) {
    // Use arrayfire default for one dimension which may be optimized
    // TODO: update this? stddev is deprecated.
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::stdev(toArray(input), biasMode, axes[0]), keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
  return this->sqrt(this->var(input, axes, /* bias = */ bias, keepDims));
}

Tensor ArrayFireBackend::norm(
    const Tensor& input,
    const std::vector<int>& axes,
    double p /* = 2 */,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // TODO: update to af::norm<af::array> if device-side specialization is
    // available. Either that or use the all-axis specializations with the below
    // implementation
    auto result = af::pow(af::abs(af::flat(toArray(input))), p);
    // Replace with af::sum<af::array>
    result = af::sum(af::sum(af::sum(result)));
    result = af::pow(result, 1 / p);
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(result), /* numDims = */ 0);
  } else {
    auto result = af::pow(af::abs(toArray(input)), p);
    result = afReduceAxes(result, axes, af::sum, keepDims);
    result = af::pow(result, 1 / p);
    return toTensor<ArrayFireTensor>(
        std::move(result),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::countNonzero(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  auto& arr = toArray(input);
  unsigned numDims;
  af::array out;
  if (isAllAxisReduction(input, axes)) {
    out = detail::condenseIndices(
        af::sum(af::sum(af::sum(af::count(arr)))), keepDims);
    numDims = 0;
  } else if (axes.size() == 1) {
    out = af::count(arr, axes.front());
    numDims = getReducedNumDims(input.ndim(), axes.size(), keepDims);
  } else {
    out = afReduceAxes(
        af::count(arr, axes.front()),
        std::vector<int>(axes.begin() + 1, axes.end()),
        af::sum,
        keepDims);
    numDims = getReducedNumDims(input.ndim(), axes.size(), keepDims);
  }
  return toTensor<ArrayFireTensor>(
      detail::condenseIndices(out, keepDims), numDims);
}

Tensor ArrayFireBackend::any(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::anyTrue<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::anyTrue(af::anyTrue(af::anyTrue(af::anyTrue(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    return toTensor<ArrayFireTensor>(
        afReduceAxes(toArray(input), axes, af::anyTrue, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}

Tensor ArrayFireBackend::all(
    const Tensor& input,
    const std::vector<int>& axes,
    const bool keepDims) {
  if (isAllAxisReduction(input, axes)) {
    // Reduce along all axes returning a singleton tensor
    // TODO: modify this to af::allTrue<af::array> to take advantage of the
    // ArrayFire reduce_all kernels once available
    return toTensor<ArrayFireTensor>(
        detail::condenseIndices(
            af::allTrue(af::allTrue(af::allTrue(af::allTrue(toArray(input)))))),
        /* numDims = */ 0);
  } else {
    return toTensor<ArrayFireTensor>(
        afReduceAxes(toArray(input), axes, af::allTrue, keepDims),
        getReducedNumDims(input.ndim(), axes.size(), keepDims));
  }
}
} // namespace fl
