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
  int nsamples = 500;
  int categories = 3;
  int feature_dim = 10;
  af::array data = af::randu(feature_dim, nsamples * categories) * 2;
  af::array label = af::constant(0.0, nsamples * categories);
  for (int i = 1; i < categories; i++) {
    data.cols(i * nsamples, (i + 1) * nsamples - 1) =
        data.cols(i * nsamples, (i + 1) * nsamples - 1) + 2 * i;
    label.rows(i * nsamples, (i + 1) * nsamples - 1) =
        label.rows(i * nsamples, (i + 1) * nsamples - 1) + i;
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
    sgd.zeroGrad();
    l.backward();

    /* Update parameters */
    sgd.step();
  }
  auto e = af::timer::stop(s);

  /* Evaluate */
  model.eval();
  result = model(input(in_));
  l = criterion(result, noGrad(out_));
  auto loss = l.array();
  af_print(loss);
  af::array max_value, prediction;
  af::max(max_value, prediction, result.array(), 0);
  auto accuracy = mean(prediction == af::transpose(label));
  af_print(accuracy);
  printf("Time/iteration: %.5f msec\n", e * 1000.0 / (nepochs - warmup_epochs));
}
