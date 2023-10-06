/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/fl/tensor/Index.h"

#include <array>
#include <iostream>
#include <memory>
#include <string>

using namespace fl;

int main(int argc, const char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " [--adam | --rmsprop]\n";
    return 1;
  }
  fl::init();

  int optim_mode = 0;
  std::string optimizer_arg = std::string(argv[1]);
  if (optimizer_arg == "--adam") {
    optim_mode = 1;
  } else if (optimizer_arg == "--rmsprop") {
    optim_mode = 2;
  }

  const int inputSize = 2;
  const int outputSize = 1;
  const double lr = 0.01;
  const double mu = 0.1;
  const int numSamples = 4;

  std::array<float, 8> hInput = {1, 1, 0, 0, 1, 0, 0, 1};
  std::array<float, 4> hOutput = {1, 0, 1, 1};

  auto in = Tensor::fromBuffer(
      {inputSize, numSamples}, hInput.data(), MemoryLocation::Host);
  auto out = Tensor::fromBuffer(
      {outputSize, numSamples}, hOutput.data(), MemoryLocation::Host);

  Sequential model;

  model.add(Linear(inputSize, outputSize));
  model.add(Sigmoid());

  auto loss = MeanSquaredError();

  std::unique_ptr<FirstOrderOptimizer> optim;

  if (optimizer_arg == "--rmsprop") {
    optim = std::make_unique<RMSPropOptimizer>(model.params(), lr);
  } else if (optimizer_arg == "--adam") {
    optim = std::make_unique<AdamOptimizer>(model.params(), lr);
  } else {
    optim = std::make_unique<SGDOptimizer>(model.params(), lr, mu);
  }

  Variable result, l;
  for (int i = 0; i < 1000; i++) {
    for (int j = 0; j < numSamples; j++) {
      model.train();
      optim->zeroGrad();

      Tensor in_j = in(fl::span, j);
      Tensor out_j = out(fl::span, j);

      // Forward propagation
      result = model(input(in_j));

      // Calculate loss
      l = loss(result, noGrad(out_j));

      // Backward propagation
      l.backward();

      // Update parameters
      optim->step();
    }

    if ((i + 1) % 100 == 0) {
      model.eval();

      // Forward propagation
      result = model(input(in));

      // Calculate loss
      // TODO: Use loss function
      Tensor diff = out - result.tensor();
      std::cout << "Average Error at iteration (" << i + 1
                << ") : " << fl::mean(fl::abs(diff)).scalar<float>() << "\n";
      std::cout << "Predicted\n"
                << result.tensor() << std::endl
                << "Expected\n"
                << out << std::endl;
    }
  }
  return 0;
}
