/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/amp/DynamicScaler.h"
#include "flashlight/ext/amp/Utils.h"
#include "flashlight/fl/flashlight.h"

namespace fl {
namespace ext {

DynamicScaler::DynamicScaler(
    double initFactor,
    double maxFactor,
    unsigned int updateInterval)
    : maxScaleFactor_(maxFactor), updateInterval_(updateInterval) {
  scaleFactor_ = fl::Variable(af::constant(initFactor, 1, 1, 1, 1, f32), false);
  flag_ = af::constant(0, 1, 1, 1, 1, s32);
}

fl::Variable DynamicScaler::scale(const fl::Variable& loss) {
  // Force casting to fp32 to avoid overflow in scaling.
  auto scaledLoss = loss.as(af::dtype::f32);
  scaledLoss = scaleLoss(scaledLoss, scaleFactor_);
  return scaledLoss;
}

bool DynamicScaler::unscale(std::vector<fl::Variable>& params) {
  for (auto& p : params) {
    validityCheck(p.grad().array(), flag_);
  }
  for (auto& p : params) {
    scaleGrads(p.grad().array(), scaleFactor_.array(), flag_);
  }
  ++successCounter_;
  return adjustScaleFactor(scaleFactor_.array(), flag_);
}

void DynamicScaler::update() {
  if (successCounter_ == updateInterval_) {
    scaleFactor_.array() = scaleFactor_.array() * 2;
    FL_VLOG(2) << "AMP: Scale factor doubled";
    successCounter_ = 0;
  } else {
    scaleFactor_.array() = scaleFactor_.array() + 2;
    FL_VLOG(3) << "AMP: Scale factor incremented";
  }
}

double DynamicScaler::getScaleFactor() const {
  return scaleFactor_.scalar<float>();
}

} // namespace ext
} // namespace fl
