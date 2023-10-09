/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/optim/Optimizers.h"

#include <cmath>

using std::vector;

// Comparision between various update rules:
// https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM

namespace fl {

FirstOrderOptimizer::FirstOrderOptimizer(
    const vector<Variable>& parameters,
    double learningRate)
    : parameters_(parameters.begin(), parameters.end()), lr_(learningRate) {}

void FirstOrderOptimizer::zeroGrad() {
  for (auto& parameter : parameters_) {
    parameter.zeroGrad();
  }
}

} // namespace fl
