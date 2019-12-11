/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "flashlight/optim/Optimizers.h"

#include <cmath>

using std::vector;

// References:
// SGD and Momentum: http://cs231n.github.io/neural-networks-3/#sgd
// Adam: https://arxiv.org/pdf/1412.6980.pdf
// RMSProp: https://arxiv.org/pdf/1308.0850v5.pdf

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
