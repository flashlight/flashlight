/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/amp/DynamicScaler.h"
#include "flashlight/ext/amp/Utils.h"
#include "flashlight/fl/flashlight.h"

namespace fl {
namespace ext {

DynamicScaler::DynamicScaler(
    float initFactor,
    float maxScaleFactor,
    unsigned int updateInterval)
    : updateInterval_(updateInterval) {
  scaleFactor_ = fl::Variable(af::constant(initFactor, 1, 1, 1, 1, f32), false);
  isInvalidArray_ = af::constant(0, 1, 1, 1, 1, s32);
  minScaleFactor_ =
      af::constant(fl::kAmpMinimumScaleFactorValue, 1, 1, 1, 1, f32);
  maxScaleFactor_ = af::constant(maxScaleFactor, 1, 1, 1, 1, f32);
}

fl::Variable DynamicScaler::scale(const fl::Variable& loss) {
  // Force casting to fp32 to avoid overflow in scaling.
  auto scaledLoss = loss.as(af::dtype::f32);
  scaledLoss = scaleLoss(scaledLoss, scaleFactor_);
  return scaledLoss;
}

bool DynamicScaler::unscale(std::vector<fl::Variable>& params) {
  for (auto& p : params) {
    validityCheck(p.grad().array(), isInvalidArray_);
  }
  for (auto& p : params) {
    scaleGrads(p.grad().array(), scaleFactor_.array(), isInvalidArray_);
  }
  ++successCounter_;
  return decreaseScaleFactor(
      scaleFactor_.array(), isInvalidArray_, minScaleFactor_);
}

void DynamicScaler::update() {
  if (successCounter_ == updateInterval_) {
    increaseScaleFactor(
        scaleFactor_.array(),
        maxScaleFactor_,
        fl::ext::ScaleFactorIncreaseForm::MULTIPLICATIVE);
    successCounter_ = 0;
  } else {
    increaseScaleFactor(
        scaleFactor_.array(),
        maxScaleFactor_,
        fl::ext::ScaleFactorIncreaseForm::ADDITIVE);
  }
}

double DynamicScaler::getScaleFactor() const {
  return scaleFactor_.scalar<float>();
}

} // namespace ext
} // namespace fl
