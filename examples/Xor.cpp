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

#include "flashlight/common/CppBackports.h"
#include "flashlight/nn/nn.h"
#include "flashlight/optim/optim.h"

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

  auto in = af::array(inputSize, numSamples, hInput.data());
  auto out = af::array(outputSize, numSamples, hOutput.data());

  Sequential model;

  model.add(Linear(inputSize, outputSize));
  model.add(Sigmoid());

  auto loss = MeanSquaredError();

  std::unique_ptr<FirstOrderOptimizer> optim;

  if (optimizer_arg == "--rmsprop") {
    optim = cpp::make_unique<RMSPropOptimizer>(model.params(), lr);
  } else if (optimizer_arg == "--adam") {
    optim = cpp::make_unique<AdamOptimizer>(model.params(), lr);
  } else {
    optim = cpp::make_unique<SGDOptimizer>(model.params(), lr, mu);
  }

  Variable result, l;
  for (int i = 0; i < 1000; i++) {
    for (int j = 0; j < numSamples; j++) {
      model.train();
      optim->zeroGrad();

      af::array in_j = in(af::span, j);
      af::array out_j = out(af::span, j);

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
      af::array diff = out - result.array();
      std::cout << "Average Error at iteration (" << i + 1
                << ") : " << af::mean<float>(af::abs(diff)) << "\n";
      std::cout << "Predicted\n";
      af_print(result.array());
      std::cout << "Expected\n";
      af_print(out);
      std::cout << "\n\n";
    }
  }
  return 0;
}
