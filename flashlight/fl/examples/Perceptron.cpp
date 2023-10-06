/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/dataset/datasets.h"
#include "flashlight/fl/meter/meters.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBase.h"

using namespace fl;

int main() {
  fl::init();

  // Create dataset
  const int nSamples = 10000;
  const int nFeat = 10;
  auto X = fl::rand({nFeat, nSamples}) + 1; // X elements in [1, 2]
  auto Y = /* signal */ fl::transpose(fl::sum(fl::power(X, 3), {0})) +
      /* noise */ fl::sin(2 * M_PI * fl::rand({nSamples}));
  // Create Dataset to simplify the code for iterating over samples
  TensorDataset data({X, Y});
  const int inputIdx = 0, targetIdx = 1;

  // Model definition - 2-layer Perceptron with ReLU activation
  Sequential model;
  model.add(Linear(nFeat, 100));
  model.add(ReLU());
  model.add(Linear(100, 1));
  // MSE loss
  auto loss = MeanSquaredError();

  // Optimizer definition
  const float learningRate = 0.0001;
  const float momentum = 0.9;
  auto sgd = SGDOptimizer(model.params(), learningRate, momentum);

  // Meter definition
  AverageValueMeter meter;

  // Start training

  std::cout << "[Multi-layer Perceptron] Started..." << std::endl;

  const int nEpochs = 100;
  for (int e = 1; e <= nEpochs; ++e) {
    meter.reset();
    for (auto& sample : data) {
      sgd.zeroGrad();

      // Forward propagation
      auto result = model(input(sample[inputIdx]));

      // Calculate loss
      auto l = loss(result, noGrad(sample[targetIdx]));

      // Backward propagation
      l.backward();

      // Update parameters
      sgd.step();

      meter.add(l.scalar<float>());
    }
    std::cout << "Epoch: " << e << " Mean Squared Error: " << meter.value()[0]
              << std::endl;
  }
  std::cout << "[Multi-layer Perceptron] Done!" << std::endl;
  return 0;
}
