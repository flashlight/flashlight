/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <iostream>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBase.h"

int main() {
  fl::init();

  // Create data
  const int nSamples = 10000;
  const int nFeat = 10;
  auto X = fl::rand({nFeat, nSamples}) + 1; // X elements in [1, 2]
  auto Y = /* signal */ fl::transpose(fl::sum(fl::power(X, 3), {0})) +
      /* noise */ fl::sin(2 * M_PI * fl::rand({nSamples}));

  // Training params
  const int nEpochs = 100;
  const float learningRate = 0.001;
  auto weight = fl::Variable(fl::rand({1, nFeat}), true /* isCalcGrad */);
  auto bias = fl::Variable(fl::full({1}, 0.0), true /* isCalcGrad */);

  std::cout << "[Linear Regression] Started..." << std::endl;

  for (int e = 1; e <= nEpochs; ++e) {
    fl::Tensor error = fl::full({1}, 0);
    for (int i = 0; i < nSamples; ++i) {
      auto input = fl::Variable(X(fl::span, i), false /* isCalcGrad */);
      auto yPred = fl::matmul(weight, input) + bias;

      auto yTrue = fl::Variable(Y(i), false /* isCalcGrad */);

      // Mean Squared Error
      auto loss = ((yPred - yTrue) * (yPred - yTrue)) / nSamples;

      // Compute gradients using backprop
      loss.backward();

      // Update the weight and bias
      weight.tensor() = weight.tensor() - learningRate * weight.grad().tensor();
      bias.tensor() = bias.tensor() - learningRate * bias.grad().tensor();

      // clear the gradients for next iteration
      weight.zeroGrad();
      bias.zeroGrad();

      error += loss.tensor();
    }

    std::cout << "Epoch: " << e
              << " Mean Squared Error: " << error.scalar<float>() << std::endl;
  }

  std::cout << "[Linear Regression] Done!" << std::endl;

  return 0;
}
