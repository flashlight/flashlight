/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/optim/Utils.h"

#include <cmath>

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

double clipGradNorm(const std::vector<Variable>& parameters, double maxNorm) {
  double gradNorm = 0.0;
  for (const auto& p : parameters) {
    if (!p.isGradAvailable()) {
      continue;
    }
    const auto& grad = p.grad().tensor();
    gradNorm += fl::sum(grad * grad).asScalar<double>();
  }
  gradNorm = std::sqrt(gradNorm);
  double scale = maxNorm / (gradNorm + 1e-6);
  if (scale >= 1.0) {
    return gradNorm;
  }
  for (auto& p : parameters) {
    if (!p.isGradAvailable()) {
      continue;
    }
    p.grad().tensor() *= scale;
  }
  return gradNorm;
}

} // namespace fl
