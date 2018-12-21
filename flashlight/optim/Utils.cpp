/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Utils.h"

namespace fl {

double clipGradNorm(const std::vector<Variable>& parameters, double max_norm) {
  double grad_norm = 0.0;
  for (const auto& p : parameters) {
    if (!p.isGradAvailable()) {
      continue;
    }
    const auto& grad = p.grad().array();
    grad_norm += std::pow(af::norm(af::flat(grad)), 2);
  }
  grad_norm = std::sqrt(grad_norm);
  double scale = (max_norm / grad_norm);
  if (scale < 1.0) {
    for (auto& p : parameters) {
      if (!p.isGradAvailable()) {
        continue;
      }
      auto& grad = p.grad().array();
      grad = scale * grad;
      p.grad().array() = grad;
    }
  }
  return grad_norm;
}

} // namespace fl
