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
  int nsamples = 100;
  int categories = 3;
  int feature_dim = 10;
  Tensor data =
      fl::rand({feature_dim, 2 * nsamples * (categories - 1), /* B = */ 1}) *
          5 +
      1;
  Tensor label = fl::full({2 * nsamples * (categories - 1), /* B = */ 1}, 0.0);
  for (int i = 1; i < categories; i++) {
    int start = (categories - 2 + i) * nsamples;
    int end = start + nsamples;
    data(i, fl::range(start, end)) = 0 - data(i, fl::range(start, end));
    label(fl::range(start, end)) = label(fl::range(start, end)) + i;
  }

  Sequential model;
  model.add(Linear(feature_dim, feature_dim));

  std::vector<int> cutoff = {1, categories};
  auto asActivation = std::make_shared<AdaptiveSoftMax>(feature_dim, cutoff);

  AdaptiveSoftMaxLoss criterion(asActivation);
  auto sgd_m = SGDOptimizer(model.params(), 1e-2);
  auto sgd_c = SGDOptimizer(criterion.params(), 1e-2);

  Variable result, l;
  int nepochs = 500, warmup_epochs = 10;
  model.train();
  criterion.train();

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
    sgd_m.zeroGrad();
    sgd_c.zeroGrad();
    l.backward();

    /* Update parameters */
    sgd_m.step();
    sgd_c.step();
  }
  auto e = fl::Timer::stop(s);

  // loss
  model.eval();
  result = model(input(in_));
  l = criterion(result, noGrad(out_));
  auto loss = l.tensor();
  std::cout << "Loss: " << loss << std::endl;

  // accuracy
  auto log_prob = criterion.getActivation()->forward(result).tensor();
  Tensor max_value, prediction;
  fl::max(max_value, prediction, log_prob, 0);
  auto accuracy = mean(prediction == label(fl::span, fl::range(0, 1)), {0});
  std::cout << "Accuracy: " << accuracy << std::endl;

  auto pred = asActivation->predict(result).tensor();
  accuracy = mean(
      fl::reshape(pred, label.shape()) == label(fl::span, fl::range(0, 1)),
      {0});
  std::cout << "Accuracy: " << accuracy << std::endl;

  // time
  fmt::print("Time/iteration: {:.5f} msec\n", e * 1000.0 / (nepochs - warmup_epochs));
}
