/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/nn/nn.h"
#include "flashlight/optim/optim.h"

using namespace fl;

int main(int /* unused */, const char** /* unused */) {
  int nsamples = 100;
  int categories = 3;
  int feature_dim = 10;
  af::array data =
      af::randu(feature_dim, 2 * nsamples * (categories - 1)) * 5 + 1;
  af::array label = af::constant(0.0, 2 * nsamples * (categories - 1));
  for (int i = 1; i < categories; i++) {
    int start = (categories - 2 + i) * nsamples;
    int end = start + nsamples - 1;
    data(i, af::seq(start, end)) = 0 - data(i, af::seq(start, end));
    label.rows(start, end) = label.rows(start, end) + i;
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

  af::array in_ = data;
  af::array out_ = label;
  af::timer s;
  for (int i = 0; i < nepochs; i++) {
    if (i == warmup_epochs) {
      s = af::timer::start();
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
  auto e = af::timer::stop(s);

  // loss
  model.eval();
  result = model(input(in_));
  l = criterion(result, noGrad(out_));
  auto loss = l.array();
  af_print(loss);

  // accuracy
  auto log_prob = criterion.getActivation()->forward(result).array();
  af::array max_value, prediction;
  af::max(max_value, prediction, log_prob, 0);
  auto accuracy = mean(prediction == af::transpose(label));
  af_print(accuracy);

  auto pred = asActivation->predict(result).array();
  accuracy = mean(pred == af::transpose(label));
  af_print(accuracy);

  // time
  printf("Time/iteration: %.5f msec\n", e * 1000.0 / (nepochs - warmup_epochs));
}
