/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <iostream>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/common.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Random.h"

using namespace fl;

#define TIME(FUNC)                                           \
  std::cout << "Timing " << #FUNC << " ...  " << std::flush; \
  std::cout << std::setprecision(5) << FUNC() * 1000.0 << " msec" << std::endl;

double timeit(std::function<void()> fn) {
  // warmup
  for (int i = 0; i < 10; ++i) {
    fn();
  }
  fl::sync();

  int num_iters = 100;
  fl::sync();
  auto start = fl::Timer::start();
  for (int i = 0; i < num_iters; i++) {
    fn();
  }
  fl::sync();
  return fl::Timer::stop(start) / num_iters;
}

double optloop(FirstOrderOptimizer& opt, const Variable& w) {
  auto input = Variable(fl::randn({10, 10}), false);
  auto fn = [&]() {
    for (int it = 0; it < 100; it++) {
      opt.zeroGrad();
      auto loss = fl::matmul(w, input);
      loss.backward();
      opt.step();
    }
  };
  return timeit(fn);
}

double sgd() {
  auto w = Variable(fl::randn({1, 10}), true);
  auto opt = SGDOptimizer({w}, 1e-3);
  return optloop(opt, w);
}

double adam() {
  auto w = Variable(fl::randn({1, 10}), true);
  auto opt = AdamOptimizer({w}, 1e-3);
  return optloop(opt, w);
}

double rmsprop() {
  auto w = Variable(fl::randn({1, 10}), true);
  auto opt = RMSPropOptimizer({w}, 1e-3);
  return optloop(opt, w);
}

double adadelta() {
  auto w = Variable(fl::randn({1, 10}), true);
  auto opt = AdadeltaOptimizer({w});
  return optloop(opt, w);
}

double nag() {
  auto w = Variable(fl::randn({1, 10}), true);
  auto opt = NAGOptimizer({w}, 1e-3);
  return optloop(opt, w);
}

int main() {
  fl::init();
  TIME(sgd);
  TIME(nag);
  TIME(adam);
  TIME(rmsprop);
  TIME(adadelta);
  return 0;
}
