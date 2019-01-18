/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>

#include "flashlight/flashlight.h"

int main() {
  af::info();

  // Create data
  const int nSamples = 10000;
  const int nFeat = 10;
  auto X = af::randu(nFeat, nSamples) + 1; // X elements in [1, 2]
  auto Y = /* signal */ af::sum(af::pow(X, 3), 0).T() +
      /* noise */ af::sin(2 * M_PI * af::randu(nSamples));

  // Training params
  const int nEpochs = 100;
  const float learningRate = 0.001;
  auto weight = fl::Variable(af::randu(1, nFeat), true /* isCalcGrad */);
  auto bias = fl::Variable(af::constant(0.0, 1), true /* isCalcGrad */);

  std::cout << "[Linear Regression] Started..." << std::endl;

  for (int e = 1; e <= nEpochs; ++e) {
    af::array error = af::constant(0, 1);
    for (int i = 0; i < nSamples; ++i) {
      auto input = fl::Variable(X(af::span, i), false /* isCalcGrad */);
      auto yPred = fl::matmul(weight, input) + bias;

      auto yTrue = fl::Variable(Y(i), false /* isCalcGrad */);

      // Mean Squared Error
      auto loss = ((yPred - yTrue) * (yPred - yTrue)) / nSamples;

      // Compute gradients using backprop
      loss.backward();

      // Update the weight and bias
      weight.array() = weight.array() - learningRate * weight.grad().array();
      bias.array() = bias.array() - learningRate * bias.grad().array();

      // clear the gradients for next iteration
      weight.zeroGrad();
      bias.zeroGrad();

      error += loss.array();
    }

    std::cout << "Epoch: " << e
              << " Mean Squared Error: " << error.scalar<float>() << std::endl;
  }

  std::cout << "[Linear Regression] Done!" << std::endl;

  return 0;
}
