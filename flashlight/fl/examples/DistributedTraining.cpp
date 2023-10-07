/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/dataset/datasets.h"
#include "flashlight/fl/distributed/distributed.h"
#include "flashlight/fl/meter/meters.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/fl/tensor/Random.h"

using namespace fl;

int main() {
  fl::init();

  fl::distributedInit(
      fl::DistributedInit::MPI,
      -1, // worldRank - unused. Automatically derived from `MPI_Comm_Rank`
      -1, // worldRank - unused. Automatically derived from `MPI_Comm_Size`
      {{fl::DistributedConstants::kMaxDevicePerNode, "8"}} // param
  );

  auto worldSize = fl::getWorldSize();
  auto worldRank = fl::getWorldRank();
  bool isMaster = (worldRank == 0);
  fl::setSeed(worldRank);

  auto reducer = std::make_shared<fl::CoalescingReducer>(
      /*scale=*/1.0 / worldSize,
      /*async=*/true,
      /*contiguous=*/true);

  // Create dataset
  const int nSamples = 10000 / worldSize;
  const int nFeat = 10;
  auto X = fl::rand({nFeat, nSamples}) + 1; // X elements in [1, 2]
  auto Y = /* signal */ fl::transpose(fl::sum(fl::power(X, 3), {0})) +
      /* noise */ fl::sin(2 * M_PI * fl::rand({nSamples}));
  // Create Dataset to simplify the code for iterating over samples
  TensorDataset data({X, Y});

  const int inputIdx = 0, targetIdx = 1;

  // Model definition - 2-layer Perceptron with ReLU activation
  auto model = std::make_shared<Sequential>();
  model->add(Linear(nFeat, 100));
  model->add(ReLU());
  model->add(Linear(100, 1));
  // MSE loss
  auto loss = MeanSquaredError();

  // synchronize parameters of the model so that the parameters in each process
  // is the same
  fl::allReduceParameters(model);

  // Add a hook to synchronize gradients of model parameters as they are
  // computed
  fl::distributeModuleGrads(model, reducer);

  // Optimizer definition
  const float learningRate = 0.0001;
  const float momentum = 0.9;
  auto sgd = SGDOptimizer(model->params(), learningRate, momentum);

  // Meter definition
  AverageValueMeter meter;

  // Start training

  if (isMaster) {
    std::cout << "[Multi-layer Perceptron] Started..." << std::endl;
  }
  const int nEpochs = 100;
  for (int e = 1; e <= nEpochs; ++e) {
    meter.reset();
    for (auto& sample : data) {
      sgd.zeroGrad();

      // Forward propagation
      auto result = model->forward(input(sample[inputIdx]));

      // Calculate loss
      auto l = loss(result, noGrad(sample[targetIdx]));

      // Backward propagation
      l.backward();
      reducer->finalize();

      // Update parameters
      sgd.step();

      meter.add(l.scalar<float>());
    }

    auto mse = meter.value();
    auto mseArr = Tensor::fromBuffer({1}, mse.data(), MemoryLocation::Host);

    fl::allReduce(mseArr);
    if (isMaster) {
      std::cout << "Epoch: " << e << " Mean Squared Error: "
                << mseArr.scalar<double>() / worldSize << std::endl;
    }
  }
  if (isMaster) {
    std::cout << "[Multi-layer Perceptron] Done!" << std::endl;
  }
  return 0;
}
