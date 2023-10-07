/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/runtime/amp/DynamicScaler.h"

#include "flashlight/fl/flashlight.h"

namespace fl::pkg::runtime {

DynamicScaler::DynamicScaler(
    double initFactor,
    double maxFactor,
    unsigned int updateInterval)
    : scaleFactor_(initFactor),
      maxScaleFactor_(maxFactor),
      updateInterval_(updateInterval) {}

fl::Variable DynamicScaler::scale(const fl::Variable& loss) {
  // Force casting to fp32 to avoid overflow in scaling.
  auto scaledLoss = loss.astype(fl::dtype::f32);
  scaledLoss = scaledLoss * scaleFactor_;
  return scaledLoss;
}

bool DynamicScaler::unscale(std::vector<fl::Variable>& params) {
  for (auto& p : params) {
    if (!p.isGradAvailable()) {
      // Add a dummy grad for params not used in the backwards pass
      p.addGrad(Variable(fl::full(p.shape(), 0., p.type()), false));
    }
    p.grad() = p.grad() / scaleFactor_;
    if (fl::isInvalidArray(p.grad().tensor())) {
      if (scaleFactor_ >= fl::kAmpMinimumScaleFactorValue) {
        scaleFactor_ = scaleFactor_ / 2.0f;
        FL_LOG(LogLevel::INFO)
            << "AMP: Scale factor decreased. New value:\t" << scaleFactor_;
      } else {
        FL_LOG(LogLevel::FATAL)
            << "Minimum loss scale reached: " << fl::kAmpMinimumScaleFactorValue
            << " with over/underflowing gradients. Lowering the "
            << "learning rate, using gradient clipping, or "
            << "increasing the batch size can help resolve "
            << "loss explosion.";
      }
      successCounter_ = 0;
      return false;
    }
  }

  ++successCounter_;
  return true;
}

void DynamicScaler::update() {
  if (scaleFactor_ >= maxScaleFactor_) {
    return;
  }

  if (scaleFactor_ == updateInterval_) {
    scaleFactor_ *= 2;
    FL_VLOG(2) << "AMP: Scale factor doubled. New value:\t" << scaleFactor_;
    successCounter_ = 0;
  } else {
    scaleFactor_ += 2;
    FL_VLOG(3) << "AMP: Scale factor incremented. New value\t" << scaleFactor_;
  }
}

double DynamicScaler::getScaleFactor() const {
  return scaleFactor_;
}

} // namespace fl
