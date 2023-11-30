/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/common/Timer.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include <fmt/core.h>
#include "flashlight/fl/tensor/Random.h"

using namespace fl;

int main(int /* unused */, const char** /* unused */) {
  fl::init();
  int nsamples = 500;
  int categories = 3;
  int feature_dim = 10;
  Tensor data = fl::rand({feature_dim, nsamples * categories}) * 2;
  Tensor label = fl::full({nsamples * categories}, 0.0);
  for (int i = 1; i < categories; i++) {
    data(fl::span, fl::range(i * nsamples, (i + 1) * nsamples)) =
        data(fl::span, fl::range(i * nsamples, (i + 1) * nsamples)) + 2 * i;
    label(fl::range(i * nsamples, (i + 1) * nsamples)) =
        label(fl::range(i * nsamples, (i + 1) * nsamples)) + i;
  }

  Sequential model;

  model.add(Linear(feature_dim, 10));
  model.add(WeightNorm(Linear(10, categories), 0));
  model.add(LogSoftmax());

  auto criterion = CategoricalCrossEntropy();

  auto sgd = SGDOptimizer(model.params(), 0.1);

  Variable result, l;

  /* Train */
  int nepochs = 1000, warmup_epochs = 10;
  model.train();

  const Tensor& in_ = data;
  const Tensor& out_ = label;
  fl::Timer s;
  for (int i = 0; i < nepochs; i++) {
    if (i == warmup_epochs) {
      s = fl::Timer::start();
    }

    /* Forward propagation */
    result = model(input(in_));

    /* Calculate loss */
    l = criterion(result, noGrad(out_));

    /* Backward propagation */
    sgd.zeroGrad();
    l.backward();

    /* Update parameters */
    sgd.step();
  }
  auto e = fl::Timer::stop(s);

  /* Evaluate */
  model.eval();
  result = model(input(in_));
  l = criterion(result, noGrad(out_));
  auto loss = l.tensor();
  std::cout << "Loss: " << loss << std::endl;
  Tensor max_value, prediction;
  fl::max(max_value, prediction, result.tensor(), 0);
  auto accuracy = mean(prediction == fl::transpose(label, {1, 0}), {0});
  std::cout << "Accuracy: " << accuracy << std::endl;
  fmt::print("Time/iteration: {:.5f} msec\n", e * 1000.0 / (nepochs - warmup_epochs));
}
