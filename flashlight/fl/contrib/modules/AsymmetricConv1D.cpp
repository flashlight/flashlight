/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/fl/contrib/modules/AsymmetricConv1D.h"

namespace fl {

void AsymmetricConv1D::checkParams() {
  if (xPad_ != static_cast<int>(PaddingMode::SAME) && xPad_ != 0) {
    throw std::invalid_argument(
        "AsymmetricConv1D: invalid xPad_, now supports only '0' or 'SAME' ");
  }
  if (futurePart_ < 0 || futurePart_ > 1) {
    throw std::invalid_argument(
        "AsymmetricConv1D: invalid futurePart_, should be in [0, 1]");
  }
}

AsymmetricConv1D::AsymmetricConv1D(
    int nIn,
    int nOut,
    int wx,
    int sx /*= 1 */,
    fl::detail::IntOrPadMode px /*= 0 */,
    float futurePart /* 0.5 */,
    int dx /* 1 */,
    bool bias /* true */,
    int groups /* 1 */)
    : Conv2D(nIn, nOut, wx, 1, sx, 1, px, 0, dx, 1, bias, groups),
      futurePart_(futurePart) {
  checkParams();
}

AsymmetricConv1D::AsymmetricConv1D(
    const Variable& w,
    int sx /*= 1 */,
    fl::detail::IntOrPadMode px /*= 0 */,
    float futurePart /*= 0.5 */,
    int dx /*= 1 */,
    int groups /*= 1 */)
    : Conv2D(w, sx, 1, px, 0, dx, 1, groups), futurePart_(futurePart) {
  checkParams();
}

AsymmetricConv1D::AsymmetricConv1D(
    const Variable& w,
    const Variable& b,
    int sx /*= 1 */,
    fl::detail::IntOrPadMode px /*= 0 */,
    float futurePart /*= 0.5 */,
    int dx /*= 1 */,
    int groups /*= 1 */)
    : Conv2D(w, b, sx, 1, px, 0, dx, 1, groups), futurePart_(futurePart) {
  checkParams();
}

Variable AsymmetricConv1D::forward(const Variable& input) {
  auto px =
      fl::derivePadding(input.dims(0), xFilter_, xStride_, xPad_, xDilation_);
  if (!(px >= 0)) {
    throw std::invalid_argument("invalid padding for AsymmetricConv1D");
  }
  Variable output;
  int cutPx = std::abs(2 * (0.5 - futurePart_)) * px;
  int asymmetryPx = px + cutPx;
  if (bias_) {
    output = conv2d(
        input,
        params_[0],
        params_[1],
        xStride_,
        yStride_,
        asymmetryPx,
        0,
        xDilation_,
        yDilation_,
        groups_);
  } else {
    output = conv2d(
        input,
        params_[0],
        xStride_,
        yStride_,
        asymmetryPx,
        0,
        xDilation_,
        yDilation_,
        groups_);
  }
  if (futurePart_ < 0.5) {
    output = output.rows(0, output.dims(0) - 1 - 2 * cutPx);
  } else if (futurePart_ > 0.5) {
    output = output.rows(2 * cutPx, output.dims(0) - 1);
  }
  return output;
}

std::string AsymmetricConv1D::prettyString() const {
  std::ostringstream ss;
  ss << "AsymmetricConv1D";
  ss << " (" << Conv2D::prettyString() << ")";
  return ss.str();
}

} // namespace fl
