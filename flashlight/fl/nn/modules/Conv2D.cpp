/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Conv2D.h"

#include <stdexcept>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/common/DynamicBenchmark.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace fl {

using detail::IntOrPadMode;

Conv2D::Conv2D(
    int nin,
    int nout,
    int wx,
    int wy,
    int sx,
    int sy,
    IntOrPadMode px,
    IntOrPadMode py,
    int dx,
    int dy,
    bool bias,
    int groups)
    : nIn_(nin),
      nOut_(nout),
      xFilter_(wx),
      yFilter_(wy),
      xStride_(sx),
      yStride_(sy),
      xPad_(px.padVal),
      yPad_(py.padVal),
      xDilation_(dx),
      yDilation_(dy),
      bias_(bias),
      groups_(groups) {
  initialize();
}

Conv2D::Conv2D(
    const Variable& w,
    int sx,
    int sy,
    IntOrPadMode px,
    IntOrPadMode py,
    int dx,
    int dy,
    int groups)
    : UnaryModule({w}),
      nIn_(w.dims(2)),
      nOut_(w.dims(3)),
      xFilter_(w.dims(0)),
      yFilter_(w.dims(1)),
      xStride_(sx),
      yStride_(sy),
      xPad_(px.padVal),
      yPad_(py.padVal),
      xDilation_(dx),
      yDilation_(dy),
      bias_(false),
      groups_(groups) {}

Conv2D::Conv2D(
    const Variable& w,
    const Variable& b,
    int sx,
    int sy,
    IntOrPadMode px,
    IntOrPadMode py,
    int dx,
    int dy,
    int groups)
    : UnaryModule({w, b}),
      nIn_(w.dims(2)),
      nOut_(w.dims(3)),
      xFilter_(w.dims(0)),
      yFilter_(w.dims(1)),
      xStride_(sx),
      yStride_(sy),
      xPad_(px.padVal),
      yPad_(py.padVal),
      xDilation_(dx),
      yDilation_(dy),
      bias_(true),
      groups_(groups) {
  if (b.dims(2) != w.dims(3)) {
    throw std::invalid_argument(
        "output channel dimension mismatch between Conv2D weight and bias");
  }
  if (b.elements() != b.dims(2)) {
    throw std::invalid_argument(
        "only 3rd dimension of Conv2D bias may be non-singleton");
  }
}

Variable Conv2D::forward(const Variable& input) {
  auto px = derivePadding(input.dims(0), xFilter_, xStride_, xPad_, xDilation_);
  auto py = derivePadding(input.dims(1), yFilter_, yStride_, yPad_, yDilation_);
  if (!(px >= 0 && py >= 0)) {
    throw std::invalid_argument("invalid padding for Conv2D");
  }

  if (bias_) {
    return conv2d(
        input,
        params_[0].as(input.type()),
        params_[1].as(input.type()),
        xStride_,
        yStride_,
        px,
        py,
        xDilation_,
        yDilation_,
        groups_,
        benchmarks_);
  } else {
    return conv2d(
        input,
        params_[0].as(input.type()),
        xStride_,
        yStride_,
        px,
        py,
        xDilation_,
        yDilation_,
        groups_,
        benchmarks_);
  }
}

void Conv2D::initialize() {
  int fanIn = xFilter_ * yFilter_ * nIn_ / groups_;
  auto wt = kaimingUniform(
      af::dim4(xFilter_, yFilter_, nIn_ / groups_, nOut_),
      fanIn,
      af::dtype::f32,
      true);
  if (bias_) {
    double bound = std::sqrt(1.0 / fanIn);
    auto bs =
        uniform(af::dim4(1, 1, nOut_, 1), -bound, bound, af::dtype::f32, true);
    params_ = {wt, bs};
  } else {
    params_ = {wt};
  }

  benchmarks_ = std::make_shared<detail::ConvBenchmarks>();
}

std::string Conv2D::prettyString() const {
  std::ostringstream ss;
  ss << "Conv2D";
  ss << " (" << nIn_ << "->" << nOut_ << ", " << xFilter_ << "x" << yFilter_
     << ", " << xStride_ << "," << yStride_ << ", ";
  if (xPad_ == static_cast<int>(PaddingMode::SAME)) {
    ss << "SAME";
  } else {
    ss << xPad_;
  }
  ss << ",";
  if (yPad_ == static_cast<int>(PaddingMode::SAME)) {
    ss << "SAME";
  } else {
    ss << yPad_;
  }
  ss << ", " << xDilation_ << ", " << yDilation_;
  ss << ")";

  if (bias_) {
    ss << " (with bias)";
  } else {
    ss << " (without bias)";
  }
  return ss.str();
}

} // namespace fl
